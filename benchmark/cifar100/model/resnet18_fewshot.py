import benchmark.cifar100.model.resnet18 as resnet18
import torch


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(
        labels, which_class
    )  # binary mask of labels equal to which_class
    class_mask_indices = class_mask.nonzero(
        as_tuple=False
    )  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

class Model(resnet18.Model):
    def __init__(
        self, block=resnet18.BasicBlock, num_block=[2, 2, 2, 2], num_classes=100
    ):
        super().__init__(block, num_block, num_classes)
    
    def encode(self, input):
        output = self.conv1(input)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        return output

    def forward(self, input):
        unique_labels = input['class_ids']
        
        x = torch.cat([input['support_set'], input['query_set']])
        
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        num_support = len(input['support_labels'])
        
        support_features = output[:num_support]
        query_features = output[num_support:]
        
        unique_labels = input['class_ids']
        support_labels = input['support_labels']
        support_features_label = [
            torch.index_select(
                support_features, 0, extract_class_indices(support_labels, c)
            )
            for c in range(len(unique_labels))
        ]
        support_features = [torch.mean(x, dim=0) for x in support_features_label]
        support_features = torch.stack(support_features)

        logits = cos_sim(query_features, support_features)
        cls_embeds = {int(unique_labels[i]):support_features[i] for i in range(unique_labels.shape[0])}
        return {"logits": logits, 'cls_protos': cls_embeds}




