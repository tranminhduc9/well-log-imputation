# Well-log Imputation với Conv-SAITS

Dự án nghiên cứu khôi phục dữ liệu well log bị thiếu bằng Conv-SAITS, kết hợp tích chập theo chiều sâu và self-attention.

## Phạm vi

- Dữ liệu GeoLink với 4 đường log: **GR, RHOB, NPHI, DTC**.
- Đầu vào là các đoạn **256 điểm chiều sâu × 4 log**.
- Xét 4 kiểu thiếu: **Single, Block-20, Block-100, Entire-Log**.
- So sánh **Conv-SAITS** với **LOCF, XGBoost, BRITS, SAITS**.
- Đánh giá bằng **MAE, RMSE, MAPE**, quá trình hội tụ và thời gian chạy.
- Thực hiện ablation để đánh giá đóng góp của các thành phần trong Conv-SAITS.

Dữ liệu được chia theo giếng thành train, validation và test. Sai số được tính tại các vị trí được che; checkpoint được chọn trên validation.

## Các notebook

- [finetune_hyper.ipynb](notebook/finetune_hyper.ipynb): tìm siêu tham số.
- [train_models.ipynb](notebook/train_models.ipynb): huấn luyện và đánh giá các mô hình.
- [ablation.ipynb](notebook/ablation.ipynb): thực nghiệm ablation.
- [visualize_results.ipynb](notebook/visualize_results.ipynb): vẽ biểu đồ từ kết quả đã lưu.

## Cách chạy

