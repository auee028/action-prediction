
# USER

## 소개
* 이 설명서는 사용자가 이 프로그램을 운용하기 위한 설명서입니다.

## 사전작업
* Ubuntu 16.04.6 에서 작동합니다.
* Python 2.7 에서 작동합니다.

## 설치
1. 상위 경로의 "requirements.txt" 를 python 환경에서 pip로 설치합니다.
    > pip install -r requirements.txt

2.  "action_app.py" 에서 메인 함수 부분의 앱 실행 부분에 아이피 주소와 포트 번호 값을 지정합니다.
```python
if __name__=="__main__":
    # app.run(host='ip address', port='port number in integer', threaded=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
```
   *   `'ip adress'` : 아이피 주소
   *   `'port number in integer'` : 정수형의 포트 번호

## 사용법
1.  python으로 "action_app.py" 를 실행합니다.
    > python action_app.py

2. 크롬 웹브라우저 창을 열어서 _아이피주소:포트_ 형식으로 주소를 입력합니다.

4. 카메라를 응시하고, 로지텍 카메라의 양 옆에 불이 들어오면 액션을 수행합니다.
    * 인식된 액션에 대한 결과 값이 크롬 웹브라우저 창에 출력됩니다.

## 문제해결
* 문제 발생 시 크롬 웹브라우저 창을 새로고침 하거나, 파이썬 프로그램을 'ctrl'+'c'로 종료하고 다시 실행합니다.

