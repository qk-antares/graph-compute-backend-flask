from flask import Flask
from controller.sim_api import sim_api
from controller.ged_api import ged_api
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 注册蓝图
app.register_blueprint(sim_api)
app.register_blueprint(ged_api)

if __name__ == '__main__':
    app.run(port=8123)
