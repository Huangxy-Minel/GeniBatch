import numpy as np
from federatedml.util import LOGGER

def __apply_cal_gradient_hp(data, fixed_point_encoder):
    fore_gradient = data[1]
    feature = np.array(data[0])
    if fixed_point_encoder:
        feature = fixed_point_encoder.encode(feature)
    LOGGER.debug(f"feature shape: {feature.shape}")
    LOGGER.debug(f"fore_gradient shape: {fore_gradient.get_shape()}")
    # start_time = time.time()
    gradient = fore_gradient.r_dot(np.ascontiguousarray(feature.transpose()))
    # end_time = time.time()
    # LOGGER.debug(f"dot time is: {end_time - start_time}")
    all_g = fixed_point_encoder.decode(gradient)
    return all_g

class HeteroGradient(object):

    @staticmethod
    def local_compute_gradient(data_instances, fore_gradient, fit_intercept, fixed_point_encoder):
        if isinstance(data_instances, list):
            # high performance mode: data_instances is list of features, fore_gradient is PEN_store
            feature_num = len(data_instances[0].features)
            data_count = len(data_instances)

            if data_count * feature_num > 100:
                LOGGER.debug("Use apply partitions")
                feat_join_grad = (data_instances, fore_gradient)
                gradient_sum = __apply_cal_gradient_hp(feat_join_grad, fixed_point_encoder)
                
                if fit_intercept:
                    # bias_grad = np.sum(fore_gradient)
                    bias_grad = fore_gradient.sum()
                    gradient_sum = gradient_sum.cat(bias_grad, 1)
                gradient = gradient_sum / data_count

            # else:
            #     LOGGER.debug(f"Original_method")
            #     feat_join_grad = (data_instances, fore_gradient)
            #     gradient_partition = self.__compute_partition_gradient_hp(feat_join_grad, 
            #                                             fit_intercept=fit_intercept,
            #                                             is_sparse=is_sparse)

            #     gradient = gradient_partition / data_count
        else:
            raise NotImplementedError("data count * feature_num <= 100!")
        return gradient