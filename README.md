
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
