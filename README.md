# Train Trading Agent


## 환경설정
- [Anaconda 3.7+](https://www.anaconda.com/distribution/)
- [TensorFlow 2.7.0](https://www.tensorflow.org/)
  - `pip install tensorflow==2.7.0`
- [plaidML](https://plaidml.github.io/plaidml/)
  - `pip install plaidml-keras==0.7.0`
  - `pip install mplfinance`
- [PyTorch](https://pytorch.org/)

# 개발 환경

- Python 3.6+
- PyTorch 1.10.1
- TensorFlow 2.7.0
- Keras 2.7.0 (TensorFlow에 포함되어 있음)

# conda 환경에서 TF 설치

## TF 1.15

> TF 1.15 사용을 위해서 Python 3.6을 설치한다.
> TF 1.15 사용할 경우 cuda 10.0, cudnn 7.4.2 (7.3.1) 설치해야 한다.
> https://www.tensorflow.org/install/source#tested_build_configurations
> https://github.com/tensorflow/models/issues/9706

```bash
conda create -n rltrader python=3.6
conda activate rltrader
pip install tensorflow-gpu==1.15
conda install cudatoolkit=10.0
conda install cudnn=7.3.1
pip install numpy
pip install pandas
```

## TF 2.5
https://www.tensorflow.org/install/source_windows?hl=en#gpu

```bash
conda create -n rltrader2 python=3.6
conda activate rltrader2
pip install tensorflow==2.5
```

CUDA 11.2
cuDNN 8.1

PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

## PyTorch

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

# 실행

- `main.py`를 통해 Command Line에서 RLTrader 실행 가능합니다. `run_e3.cmd`를 참고해 주세요.
- RLTrader 모듈들을 임포트하여 사용하는 것도 가능합니다. `main.py` 코드를 참고해 주세요.

## 예시

```bash
python main.py --mode train --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net dnn --start_date 20180101 --end_date 20191231
```


## v3

- 종목 데이터
  - `date,open,high,low,close,volume,per,pbr,roe,open_lastclose_ratio,high_close_ratio,low_close_ratio,diffratio,volume_lastvolume_ratio,close_ma5_ratio,volume_ma5_ratio,close_ma10_ratio,volume_ma10_ratio,close_ma20_ratio,volume_ma20_ratio,close_ma60_ratio,volume_ma60_ratio,close_ma120_ratio,volume_ma120_ratio,ind,ind_diff,ind_ma5,ind_ma10,ind_ma20,ind_ma60,ind_ma120,inst,inst_diff,inst_ma5,inst_ma10,inst_ma20,inst_ma60,inst_ma120,foreign,foreign_diff,foreign_ma5,foreign_ma10,foreign_ma20,foreign_ma60,foreign_ma120`
- 시장 데이터
  - `date, market_kospi_ma5_ratio,market_kospi_ma20_ratio,market_kospi_ma60_ratio,market_kospi_ma120_ratio,bond_k3y_ma5_ratio,bond_k3y_ma20_ratio,bond_k3y_ma60_ratio,bond_k3y_ma120_ratio`

## v4

- 종목 데이터: v3 종목 데이터와 동일
- 시장 데이터: v3 시장 데이터에 다음 시장 데이터 추가
  - TBD

## etf

- 종목 데이터
  - `perb, bw, MACD_ratio, RSI,slow_k,slow_d, close 의 ma 5,10,20,60,120_ratio , open_lastclose_ratio, high_close_ratio, low_close_ratio, close_lastclose_ratio,volume_lastvolume_ratio`
- 시장 데이터
 - TBD

# 프로파일링
- `python -m cProfile -o profile.pstats main.py ...`
- `python profile.py`

# Troubleshooting

## TF 1.15에서 다음 에러가 나면 Python 3.6으로 맞춰준다.
```
NotImplementedError: Cannot convert a symbolic Tensor (lstm/strided_slice:0) to a numpy array.
```

## original_keras_version = f.attrs['keras_version'].decode('utf8') AttributeError: 'str' object has no attribute 'decode'
```
https://github.com/keras-team/keras/issues/14265
https://pypi.org/project/h5py/#history
pip install h5py==2.10.0
```
