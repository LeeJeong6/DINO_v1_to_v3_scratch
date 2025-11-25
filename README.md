
프로젝트 : DINO V1 ~V3까지 scratch부터 코딩하기
조건 : GPT안쓰기
달성 목표 : 원본 코드에 존재하는 모르는 함수 및 디테일 알아가기

# 2025/11/7
- 프로젝트 시작
- DINOv1공부
- ViT(visiontransformer코드 head 빼고 완성)
- 다음 계획 : teacher model, student model, loss 코드 완성하기 

# 2025/11/10
- DINO method공부
- ImageNet클래스로더 만들기 
- main_dino.py만들기 시작
- 다음 계획 : 데이터 augmentation 내 데이터로더에 추가해서 어떻게 되는지 직접 실험 


# 2025/11/12
- Dataaug관련 함수 및 클래스 모두 생성
- multi gpu 관련 함수 및 로직 필터링 중
- 데이터로더로 aug된 이미지 시각화
- vision_transformer.py의 head추가, apply,init_weight 등 가중치 초기화 방법들 공부
- 다음 계획 : head에 존재하는 다른 초기화 방법들 실험 및 공부, main_dino에 존재하는 utils코드 공부하기 

# 2025/11/13
- multigpu 및 가중치 초기화 등 디테일한 요소는 다 버림
- 학습이 되는 과정까지 직접 확인
- 코드 디버깅 연습 및 학습 코드까지 모두 완료
- 다음 계획 : schedular 도입, 로그 출력 원본 코드 확인하기 

# 2025/11/19
- multigpu 새팅 완성
- pt파일 load 후 벡터간 cosine similarity확인결과, collapse발생 확인
- 다음 실험 전, DDP와 util 구체화하기 

# 2025/11/24
- mixed precision 추가
- DDP 구체화 및 log 원본 코드와 똑같이 맞추기
- parser 수정

# 2025/11/25
- EMA 업데이트 코드 수정
- 원본 코드 util/클래스 추가
- val -> train으로 수정
- DDP실행방법 점검, 수정 python ~~~.py --> python -m torch.distributed.launch --nproc_per_node=4 main_dino.py

