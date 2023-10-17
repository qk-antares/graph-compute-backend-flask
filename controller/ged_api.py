from flask import Blueprint, request, jsonify
from service.ged_service import *

ged_api = Blueprint('ged_api', __name__)


@ged_api.route('/ged/compute', methods=['POST'])
def ged():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2 = data.get('graph2', {})
    paths, cost, total_time = compute_ged(graph1, graph2)
    return jsonify({
        'code': 200,
        'data': {
            'paths': {
                'nodes': paths[0][0],
                'edges': paths[0][1]
            },
            'cost': cost,
            'timeUse': total_time
        }
    })
