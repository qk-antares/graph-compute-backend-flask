from flask import Blueprint, request, jsonify
from service.sim_service import *

sim_api = Blueprint('sim_api', __name__)


@sim_api.route('/sim/compute', methods=['POST'])
def sim_score():
    # 获取POST请求中的JSON数据
    graph_json = request.get_json()
    prediction, total_time = compute_sim_score(graph_json)
    return jsonify({'code': 200, 'data':  {'simScore': prediction.item(), 'timeUse': total_time}})
