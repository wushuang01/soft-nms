def box_soft_nms(bboxes,scores,labels,nms_threshold=0.3,soft_threshold=0.3,sigma=0.5,mode='union'):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes:
    :param scores:
    :param labels:
    :param nms_threshold:
    :param soft_threshold:
    :return:
    """
    unique_labels = labels.cpu().unique().cuda()

    box_keep = []
    labels_keep = []
    scores_keep = []
    for c in unique_labels:
        c_boxes = bboxes[labels == c]
        c_scores = scores[labels == c]
        weights = c_scores.clone()
        x1 = c_boxes[:, 0]
        y1 = c_boxes[:, 1]
        x2 = c_boxes[:, 2]
        y2 = c_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = weights.sort(0, descending=True)
        while order.numel() > 0:
            i = order[0]
            box_keep.append(c_boxes[i])
            labels_keep.append(c)
            scores_keep.append(c_scores[i])

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1 + 1).clamp(min=0)
            h = (yy2 - yy1 + 1).clamp(min=0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids_t= (ovr>=nms_threshold).nonzero().squeeze()

            weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)

            ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            c_boxes = c_boxes[order[1:]][ids]
            c_scores = weights[order[1:]][ids]
            _, order = weights[order[1:]][ids].sort(0, descending=True)
            if c_boxes.dim()==1:
                c_boxes=c_boxes.unsqueeze(0)
                c_scores=c_scores.unsqueeze(0)
            x1 = c_boxes[:, 0]
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    return box_keep, labels_keep, scores_keep
