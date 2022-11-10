import dataclasses

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="boardlogger")


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
@dataclasses.dataclass
class inforun:
    current_step = 0
    lossstep = 0
    run_now = False
    epoch = 0
    traj_idx = 0


rundata = inforun()
