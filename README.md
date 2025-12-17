
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

# 2025/11/28
- eval_linear.py 만들기
- 아직 학습중이라 GPU에 올려서 실험해보지 못했음
- 학습이 다 되면 python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py
- eval_linear.py는 fc layer로 finetuning하는거라 특별한 기법은 없다. 그래서 코드만 이해 후 복붙함

# 2025/12/6
- 학습 완료된 모델로 classifier fine tuning시작
- log를 보면 더 loss가 줄어들 수는 있을 것 같은데 이게 collapse를 잘 피한건지 의문임
- eval_linear.py 오타 수정, vision_transformer.py 수정
- https://drive.google.com/file/d/1DjCuKCwuE_XIqlSvD2K04LPAvMGqDnM4/view?usp=sharing에서 다운 가능

# 2025/12/17
- 애초에 학습이 다 안됐던 것 같아서 trunc_normal, init_weights 등 가중치 관련 함수들을 추가하고 main_dino부터 다시 학습 시작
- main_dino의 clip_grad함수도 추가해서 새팅, ImageNet경로 수정
