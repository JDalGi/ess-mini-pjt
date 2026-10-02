# 원본 데이터

MIT–Stanford Battery Dataset의 원본 MATLAB v7.3(HDF5) 파일을 이 폴더에 둡니다. 원본은 별도로 준비하며, 대용량 `.mat` 파일은 `.gitignore`에 따라 Git에서 제외됩니다.

| 필요한 파일명                                          | DAY 1       | DAY 2                                            |
| ------------------------------------------------------ | ----------- | ------------------------------------------------ |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 EDA | 학습·그룹 CV·Hold-out                            |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 EDA | 최종 테스트                                      |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 EDA | 공통 로더에서 읽지만 학습·평가에는 사용하지 않음 |

파일 이름을 그대로 유지하고 `data/` 바로 아래에 배치합니다. 현재 공통 로더는 세 배치를 모두 읽으므로 세 파일이 모두 필요합니다.

코드는 셀별 수명·충전 정책·사이클 요약값, 10·100번째 사이클의 Qdlin 곡선과 cycle 10의 전류·시간을 읽습니다. 초기 예측 입력은 100사이클까지 관측 가능한 정보만 사용하며, 분석 단위는 배터리 셀 하나이며 수명 라벨이 없는 셀은 타깃 통계와 지도학습에서 제외합니다.
