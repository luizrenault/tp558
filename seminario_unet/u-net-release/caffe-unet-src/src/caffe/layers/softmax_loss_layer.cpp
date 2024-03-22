#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  class_loss_weights_.resize(this->layer_param_.loss_param().class_loss_weights_size());
  for( int i=0; i < class_loss_weights_.size(); ++i) {
    class_loss_weights_[i] = this->layer_param_.loss_param().class_loss_weights(i);
    std::cout << "class_loss_weights[" << i << "] = " << class_loss_weights_[i] << std::endl;
  }
  
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  if( bottom.size() == 3) {
    // weighted version using a third input blob (by Olaf)
    Dtype weightsum = 0;
    const Dtype* weight_data = bottom[2]->cpu_data();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.channels());
        loss -= weight_data[j] * 
            log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                         Dtype(FLT_MIN)));
        weightsum += weight_data[j];
      }
    }
    if (normalize_) {
      top[0]->mutable_cpu_data()[0] = loss / weightsum;
    } else {
      top[0]->mutable_cpu_data()[0] = loss / num;
    }

  } else if(class_loss_weights_.size() > 0) {
    // weighted version using class-wise loss weights (by Olaf)
    Dtype weightsum = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.channels());
        Dtype weight = 1;
        if( label_value < class_loss_weights_.size()) {
          weight = class_loss_weights_[label_value];
        }
        
        loss -= weight * 
            log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                         Dtype(FLT_MIN)));
        weightsum += weight;
      }
    }
    if (normalize_) {
      top[0]->mutable_cpu_data()[0] = loss / weightsum;
    } else {
      top[0]->mutable_cpu_data()[0] = loss / num;
    }
  } else {
    // standard version of SoftmaxWithLossLayer
    int count = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.channels());
        loss -= log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                             Dtype(FLT_MIN)));
        ++count;
      }
    }
    if (normalize_) {
      top[0]->mutable_cpu_data()[0] = loss / count;
    } else {
      top[0]->mutable_cpu_data()[0] = loss / num;
    }
  }
  
  
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    if( bottom.size() == 3) {
       // weighted version using a third input blob (by Olaf)
      Dtype weightsum = 0;
      const Dtype* weight_data = bottom[2]->cpu_data();
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; ++j) {
          const int label_value = static_cast<int>(label[i * spatial_dim + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] = 0;
            }
          } else {
            bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] *= weight_data[j];
            }
            weightsum += weight_data[j];
          }
        }
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      if (normalize_) {
        caffe_scal(prob_.count(), loss_weight / weightsum, bottom_diff);
      } else {
        caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
      }
    } else if(class_loss_weights_.size() > 0) {
      // weighted version using class-wise loss weights (by Olaf)
      Dtype weightsum = 0;
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; ++j) {
          const int label_value = static_cast<int>(label[i * spatial_dim + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] = 0;
            }
          } else {
            Dtype weight = 1;
            if( label_value < class_loss_weights_.size()) {
              weight = class_loss_weights_[label_value];
            }
         
            bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;

            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] *= weight;
            }
           weightsum += weight;
          }
        }
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      if (normalize_) {
        caffe_scal(prob_.count(), loss_weight / weightsum, bottom_diff);
      } else {
        caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
      }
    } else {
     // standard version of SoftmaxWithLossLayer
      int count = 0;
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; ++j) {
          const int label_value = static_cast<int>(label[i * spatial_dim + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] = 0;
            }
          } else {
            bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;
            ++count;
          }
        }
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      if (normalize_) {
        caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
      } else {
        caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
      }
    } 
    
    
  }
}

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_LOSS, SoftmaxWithLossLayer);

}  // namespace caffe
