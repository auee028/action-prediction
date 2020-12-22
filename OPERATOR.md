
# OPERATOR

## 소개
* 이 설명서는 운영자가 이 프로그램을 운용하기 위한 설명서입니다.

## 사전작업
* Ubuntu 16.04.6 LTS 에서 작동합니다.
* Python 2.7 에서 작동합니다.
* NVIDIA CUDA 8.0를 지원하는 GPU와 그래픽 드라이버를 필요로 합니다.
* CUDA 8.0과 cuDNN 6.0 에서 작동합니다.
* 로지텍 C920 PRO HD 웹캠 설치 시 사용자가 프로그램 동작을 인식하는데 용이합니다.

## 설치
1. 상위 경로의 "requirements.txt" 를 python 환경에서 pip로 설치합니다.
    > pip install -r requirements.txt

2. "action_pred.py" 에서 requests.post 및 requests.get 부분이 있는 라인에 아이피 주소 및 포트 번호 값을 지정합니다.

3. [save_model.zip](https://drive.google.com/file/d/1_gXZAU5kfqmAvjs0EAR1bA3s3IlvmMtQ/view?usp=sharing) 파일을 다운받아 상위 경로에 압축을 해제합니다.

## 사용법
1. 사용자가 필요한 프로그램을 모두 실행한 것을 확인 후, python으로 "action_pred.py" 를 실행합니다. 
    * 이 때, 설치된 카메라가  올바른 카메라 번호를 가지고 있는지 확인하고 변수 "--cam"에 올바른 카메라 번호 값을 지정하여 프로그램을 실행하도록 합니다(default = 0).
      > python action_pred.py --cam=0

2. 정상적으로 실행 시, 사용자 행동 인식 및 예측 결과가 출력되는 것을 터미널을 통해 확인할 수 있습니다.


## 문제해결
* 문제 발생 시 'ctrl'+'c' 를 입력하여 프로그램을 종료하고 다시 실행합니다.
