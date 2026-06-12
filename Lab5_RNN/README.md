# Bài thực hành Deep Learning: Recurrent Neural Network

Project này cài đặt RNN để dự báo chuỗi thời gian cho 4 bài:

1. Dự báo giá nhà với cột `price` trong `datasets/raw_sales.csv`.
2. Dự báo giá Bitcoin với cột `priceUSD` trong `datasets/BTC_DATA.csv`.
3. Dự báo điện thế tiêu thụ với cột `Voltage` trong `datasets/household_power_consumption.txt`.
4. Dự báo giá đóng cửa NIFTY với cột `close` trong `datasets/NIFTY_stock_market.csv`.

Do giáo viên không cung cấp dataset, project ưu tiên dùng dataset public tương đương và chuẩn hóa tên file/cột đúng theo đề. Nếu không tải được do mạng, Kaggle hoặc API bị chặn, script sẽ in hướng dẫn và tạo dataset demo nhỏ để sinh viên vẫn chạy được pipeline học tập.

## 1. Giới thiệu bài thực hành

Bài thực hành minh họa cách dùng Recurrent Neural Network cho dữ liệu chuỗi thời gian. Mỗi mô hình chỉ dùng một cột mục tiêu, chuẩn hóa bằng `MinMaxScaler`, tạo dữ liệu dạng sequence với `time_steps = 12`, train/test theo tỷ lệ 80/20, sau đó đánh giá bằng RMSE và MAE.

## 2. RNN là gì?

RNN là mạng nơ-ron hồi tiếp, có khả năng xử lý dữ liệu tuần tự bằng cách giữ thông tin từ các bước thời gian trước. Vì vậy RNN phù hợp với các bài toán như dự báo giá, dự báo điện năng, xử lý văn bản và chuỗi tín hiệu.

Trong project này mô hình chính là:

```python
Sequential([
    SimpleRNN(64, activation="tanh", input_shape=(time_steps, 1)),
    Dense(32, activation="relu"),
    Dense(1)
])
```

Mô hình dùng `optimizer="adam"` và `loss="mean_squared_error"`.

## 3. Cấu trúc project

```text
rnn_deeplearning_project/
|-- RNN_Thuc_Hanh_DeepLearning.ipynb
|-- app.py
|-- requirements.txt
|-- README.md
|-- datasets/
|   |-- raw_sales.csv
|   |-- BTC_DATA.csv
|   |-- household_power_consumption.txt
|   `-- NIFTY_stock_market.csv
|-- models/
|   |-- rnn_house_price.h5
|   |-- rnn_btc_price.h5
|   |-- rnn_voltage.h5
|   `-- rnn_nifty_close.h5
|-- scalers/
|   |-- scaler_house_price.pkl
|   |-- scaler_btc_price.pkl
|   |-- scaler_voltage.pkl
|   `-- scaler_nifty_close.pkl
|-- training/
|   |-- train_house_price_rnn.py
|   |-- train_btc_rnn.py
|   |-- train_voltage_rnn.py
|   `-- train_nifty_rnn.py
|-- utils/
|   |-- data_loader.py
|   |-- preprocessing.py
|   `-- prediction.py
|-- templates/
|   |-- index.html
|   `-- result.html
`-- static/
    `-- style.css
```

Các file dataset, model và scaler sẽ được tạo khi chạy script train nếu chưa có sẵn.

## 4. Dataset sử dụng

### raw_sales.csv

Nguồn ưu tiên: dataset giá nhà public tương đương. Script dùng dataset `housing.csv` public từ repository Hands-On ML nếu tải được, lấy cột `median_house_value` và đổi tên thành `price`.

Nếu muốn dùng dataset đúng kiểu House Property Sales Time Series từ Kaggle, hãy tải thủ công, đổi cột giá bán như `Price`, `SalePrice`, `sale_price` thành `price`, rồi lưu vào:

```text
datasets/raw_sales.csv
```

### BTC_DATA.csv

Nguồn ưu tiên: `BTC-USD` từ `yfinance`. Script lấy cột `Close`, đổi tên thành `priceUSD`, rồi lưu vào:

```text
datasets/BTC_DATA.csv
```

### household_power_consumption.txt

Nguồn ưu tiên: Individual Household Electric Power Consumption từ UCI:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
```

Script đọc một phần dữ liệu để train nhanh, xử lý missing value có ký hiệu `?`, và dùng cột `Voltage`.

### NIFTY_stock_market.csv

Nguồn ưu tiên: `^NSEI` từ `yfinance`. Script lấy cột `Close`, đổi tên thành `close`, rồi lưu vào:

```text
datasets/NIFTY_stock_market.csv
```

Có thể thay bằng dataset NIFTY-50 Stock Market Data public, miễn là cột cần dự báo được đổi tên thành `close`.

## 5. Cách cài thư viện

```bash
pip install -r requirements.txt
```

Nếu dùng Google Colab, upload toàn bộ thư mục project hoặc upload notebook, sau đó chạy cell cài thư viện khi cần.

## 6. Cách train từng mô hình

Chạy từ thư mục `rnn_deeplearning_project`:

```bash
python training/train_house_price_rnn.py
python training/train_btc_rnn.py
python training/train_voltage_rnn.py
python training/train_nifty_rnn.py
```

Mỗi script sẽ:

- Đọc hoặc tải dataset.
- Kiểm tra cột cần dự báo.
- Xử lý missing value.
- Chuẩn hóa dữ liệu bằng `MinMaxScaler`.
- Tạo sequence dạng `(samples, time_steps, features)`.
- Train SimpleRNN trong 20 epochs mặc định.
- Tính RMSE, MAE.
- Lưu model vào `models/`.
- Lưu scaler vào `scalers/`.
- Lưu biểu đồ loss và dự báo vào `static/training_plots/`.

## 7. Cách chạy notebook

Notebook chính:

```text
RNN_Thuc_Hanh_DeepLearning.ipynb
```

Chạy trên Google Colab:

1. Upload `RNN_Thuc_Hanh_DeepLearning.ipynb` lên Google Colab.
2. Upload thêm các thư mục `utils/`, `training/`, hoặc upload cả project.
3. Vào `Runtime > Change runtime type > GPU` nếu có.
4. Chạy từng cell theo thứ tự.

Notebook có Markdown tiếng Việt, hiển thị 5 dòng đầu của dataset, biểu đồ chuỗi thời gian, loss/val_loss, Actual vs Prediction, RMSE, MAE và bảng tổng kết.

## 8. Cách chạy Flask Web

Train ít nhất một mô hình trước, ví dụ:

```bash
python training/train_btc_rnn.py
```

Sau đó chạy web:

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nhập 12 giá trị gần nhất, cách nhau bằng dấu phẩy, ví dụ:

```text
100, 102, 105, 107, 108, 110, 112, 115, 116, 118, 120, 123
```

Nếu chưa train model hoặc chưa có scaler, web sẽ hiển thị:

```text
Chưa có model. Vui lòng chạy file train tương ứng trước.
```

## 9. Nhận xét kết quả

- RNN phù hợp với dữ liệu chuỗi thời gian vì có thể dùng thông tin từ các bước trước.
- `time_steps` càng lớn thì mô hình có nhiều thông tin quá khứ hơn nhưng thời gian train lâu hơn.
- Dữ liệu tài chính như BTC và NIFTY biến động mạnh nên dự báo khó hơn.
- Dữ liệu thiếu cần được xử lý trước khi train, đặc biệt dataset điện năng có ký hiệu missing value là `?`.
- Kết quả dự báo chỉ dùng cho mục đích học tập, không dùng để đầu tư thực tế.
