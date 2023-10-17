from model.simgnn.param_parser import parameter_parser
from model.simgnn.simgnn import SimGNNTrainer
import time

# 参数解析
args = parameter_parser()
args.load_path = './model/simgnn/model.pkl'
# 加载模型
trainer = SimGNNTrainer(args)
trainer.load()
trainer.model.eval()


def compute_sim_score(graph_json):
    data = trainer.transfer_to_torch(graph_json)
    start_time = time.time()
    prediction = trainer.model(data)
    end_time = time.time()
    total_time = end_time - start_time
    return prediction, total_time
