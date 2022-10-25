import os
import sys
import logging
import argparse
import json

from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader import data_manager

'''
python main.py --mode test --ver etf --name test_221022 --stock_code DJI --rl_method a2c --net lstm 
--start_date 20190101 --end_date 20221231 
--pretrained_value_net train_221022_a2c_lstm_value.mdl --pretrained_policy_net train_221022_a2c_lstm_policy.mdl
--save_folder test2
'''

'''  (괄호 안은 생략 가능)
python main.py (--mode train) (--ver etf) --name train_221023 --stock_code DJI --net lstm (--rl_method a2c) 
--start_date 19880115 --end_date 20181231 (--save_folder output/experiment221023)
'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'custom', 'etf'], default='etf', help='What version of Dataset will be used' )
    parser.add_argument('--name', default='--name')
    parser.add_argument('--stock_code', nargs='+', help='--stock_code 1234 2345 3456 4567')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default = 'a2c')
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')

    parser.add_argument('--start_date', default='20200101')
    parser.add_argument('--end_date', default='20201231')
    parser.add_argument('--save_folder', type=str, default = 'output', help='all logs will be saved into this PATH')

    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--discount_factor', type=float, default=0.7)
    parser.add_argument('--balance', type=int, default=100000000)

    ####### Additional Arguments ########
    parser.add_argument('--pretrained_value_net', type=str, default='' , help='PATH of pretrained model. must be ended with ".mdl", ')
    parser.add_argument('--pretrained_policy_net', type=str, default='', help='If given, load ./models/PATH.mdl. If empty, train new model')
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_epoches', type=int)
    parser.add_argument('--start_eps', type=float)
    args = parser.parse_args()

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}_{args.stock_code}_{utils.get_time_str()}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1

    # 커스텀 설정 221023 추가
    if args.num_steps :
        if args.net in ['lstm','cnn']:
            num_steps = args.num_steps
    if args.num_epoches :
        if args.mode in ['train','update']:
            num_epoches = args.num_epoches
    if args.start_eps :
        if args.mode in ['train','update']:
            start_epsilon = args.start_eps        

    

    

    
    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, args.save_folder, output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    if args.pretrained_value_net :
        value_network_name = args.pretrained_value_net
    if args.pretrained_policy_net :    
        policy_network_name = args.pretrained_policy_net

    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []


    #################################################DEBUGGING AREA#################################################
    print('\n','-'*50 , "\tDEBUGGING..  in [main.py]\t", '-'*50 )
    print(f"\targs.mode = {args.mode}  -> learning mode= {learning}")
    if args.mode in ['train', 'update']:
        print(f"\tValue_network and Policy_network will be trained and saved in folders:\n{value_network_path}, \n{policy_network_path}")
    if args.pretrained_value_net :
        print(f"\t---Loading Pretrained model ({args.pretrained_value_net}) in ./models ")
    if args.pretrained_policy_net :    
        print(f"\t---Loading Pretrained model ({args.pretrained_policy_net}) in ./models ")
    

    print(f"\tAll logs will be saved in [{output_path}]")
    #################################################DEBUGGING AREA#################################################

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()
