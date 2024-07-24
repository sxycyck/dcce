from bootstrap.lib.options import Options
from block.models.criterions.vqa_cross_entropy import VQACrossEntropyLoss
from .rubi_criterion import RUBiCriterion
from .cfvqa_criterion import CFVQACriterion
from .dcce_criterion import DCCECriterion
def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split
    eval_only = 'train' not in engine.dataset
    
    opt = Options()['model.criterion']
    if split == "test" and 'tdiuc' not in Options()['dataset.name']:
        return None
    if name == 'vqa_cross_entropy':
        criterion = VQACrossEntropyLoss()
    elif name == "rubi_criterion":
        criterion = RUBiCriterion(
            question_loss_weight=opt['question_loss_weight']
        )
    elif name == "cfvqa_criterion":
        criterion = CFVQACriterion(
            question_loss_weight=opt['question_loss_weight'],
            vision_loss_weight=opt['vision_loss_weight'],
            is_va=True
        )
    elif name == "dcce_criterion":
        criterion = DCCECriterion(
            question_loss_weight = opt['question_loss_weight'],
            entropy_loss_weight = opt['entropy_loss_weight'],
            engine = engine
        )
    else:
        raise ValueError(name)
    return criterion
