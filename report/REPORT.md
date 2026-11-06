# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Mỗi record mô tả một vật thể riêng biệt được mô hình YOLO11 phát hiện trong ảnh, không phải toàn bộ bức ảnh. Vì mỗi record chứa thông tin về lớp đối tượng (class_name), điểm tin cậy (score) và tọa độ hộp bao (bbox_xyxy) xác định vị trí của vật thể đó. Ngoài ra, trong đoạn code, vòng lặp for duyệt qua từng vật thể được phát hiện và tạo một record riêng cho mỗi vật thể.
3. Danh sách tên lớp (class list) được xác định sẵn theo taxonomy và đi kèm checkpoint của mô hình, không phải do mô hình tự nghĩ ra.
4. Cần ghi đầy đủ class_id, class_name và taxonomy_name để xác định chính xác lớp đối tượng, tránh nhầm lẫn và đảm bảo tính nhất quán của dữ liệu.
5. Guideline cần quy định rõ cách xác định đối tượng hoặc chủ đề chính của bức ảnh để chọn một lớp phù hợp, chẳng hạn dựa vào mức độ nổi bật, kích thước, vị trí và nội dung trọng tâm. Đồng thời, cần có quy tắc xử lý khi nhiều đối tượng cùng nổi bật hoặc không thể xác định một lớp đại diện, nhằm đảm bảo việc gán nhãn nhất quán và tránh lựa chọn theo cảm tính.
6. Không được lấy score của model làm ground truth vì score chỉ thể hiện mức độ tin cậy của mô hình đối với dự đoán, không đảm bảo dự đoán đó chính xác. Mô hình vẫn có thể dự đoán sai dù có score cao.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1.`class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Trong ảnh kitchen, mô hình YOLO11 phát hiện một đối tượng thuộc lớp person (người). Hộp bao (bounding box) của đối tượng này có:
- Góc trên bên trái: (385.33, 69.24) pixel.
- Góc dưới bên phải: (498.92, 348.92) pixel.
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
3. Chọn 2 ngưỡng là 0.50 và 0.80. Ở ngưỡng 0.50 có 6 prediction, ngưỡng 0.80 có 1 prediction

4. Khi hạ ngưỡng confidence, mô hình giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp. Vì vậy, reviewer phải kiểm tra nhiều bounding box hơn, xác định các dự đoán sai (false positive), hộp bị trùng lặp, sai lớp đối tượng hoặc có tọa độ không chính xác. Reviewer cần đối chiếu với ảnh gốc và guideline để quyết định hộp nào hợp lệ, hộp nào cần loại bỏ hoặc chỉnh sửa, nhằm đảm bảo chất lượng nhãn (ground truth).
5. Khi gán nhãn bằng bounding box, người gán nhãn (annotator) cần vẽ một hộp chữ nhật bao sát toàn bộ phần nhìn thấy được của từng vật thể, đảm bảo các cạnh hộp không cắt mất phần vật thể và hạn chế chứa quá nhiều vùng nền không liên quan. Mỗi vật thể được gán một bounding box riêng và phải tuân theo cùng một quy tắc để đảm bảo tính nhất quán giữa các ảnh.
6. Oven bị người che khuất. Ưu tiên đưa vào guideline quy tắc chỉ vẽ bounding box bao sát phần vật thể nhìn thấy được, không tự suy đoán phần bị che khuất. Quy tắc này có thể áp dụng cho nhiều ảnh khác nhau và giúp việc gán nhãn nhất quán. Nếu mức độ che khuất khiến không thể xác định rõ đối tượng hoặc cách vẽ hộp, annotator cần hỏi Lab Coach (escalation) để được hướng dẫn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `instance_id`="kitchen-002" ; `class_name`="bowl" ; `score`=0.735743 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[[53.0 ,344.0], [52.0 ,345.0], [50.0, 345.0]]
2. So với bounding box, polygon cung cấp thông tin chi tiết và chính xác hơn về hình dạng cũng như đường biên của vật thể. Trong khi bounding box chỉ sử dụng một hình chữ nhật để bao quanh vật thể, polygon sử dụng nhiều điểm tọa độ để mô tả đường viền thực tế, kể cả những hình dạng phức tạp hoặc không đều. Nhờ đó, polygon giúp xác định chính xác hơn vùng pixel thuộc vật thể và hạn chế bao gồm các vùng nền không liên quan.
3. `instance_id` dùng để định danh và phân biệt từng đối tượng cụ thể (instance) được phát hiện trong một ảnh, kể cả khi các đối tượng thuộc cùng một lớp.
`instance_id` không phải class ID vì class ID dùng để xác định loại đối tượng (ví dụ: người, xe, chó), trong khi instance ID dùng để phân biệt từng đối tượng riêng biệt.
`instance_id` cũng không phải tracking ID vì tracking ID dùng để theo dõi cùng một đối tượng qua nhiều khung hình trong video, còn instance ID trong bài thực hành chỉ có ý nghĩa định danh đối tượng trong bộ kết quả hiện tại, không dùng để theo dõi đối tượng theo thời gian.
4. Khi gán nhãn bằng segmentation mask, người gán nhãn (annotator) cần vẽ đường biên bám sát hình dạng thực tế của phần vật thể nhìn thấy được, đảm bảo bao phủ đầy đủ các pixel thuộc vật thể và hạn chế bao gồm vùng nền hoặc vật thể khác. Đối với những vật thể bị che khuất, chỉ vẽ mask cho phần nhìn thấy được, không tự suy đoán phần bị che, nhằm đảm bảo tính chính xác và nhất quán trong quá trình gán nhãn.
5. Dụng cụ nấu ăn ở dưới Dining table đã bị che. Sử dụng guideline để quy định rằng annotator chỉ vẽ đường biên mask theo phần vật thể thực sự nhìn thấy, không tự suy đoán phần bị che và không gộp hai vật thể chạm nhau thành một mask.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):


| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp đại diện cho toàn bộ ảnh, gồm `class_id`, `class_name` và taxonomy. | Ảnh `traffic` có nhiều đối tượng nên có thể khó xác định lớp đại diện. Cần đối chiếu với dự đoán `rank=1` trong JSON. | Xác định chủ đề chính và chọn một lớp theo guideline. | Kiểm tra nhãn có phù hợp với ảnh và taxonomy không. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và bounding box `bbox_xyxy`. | Ảnh `kitchen`: model phát hiện `person`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`. Cần kiểm tra hộp có bao sát người không. | Gán đúng lớp, vẽ bounding box bao sát phần vật thể nhìn thấy theo guideline. | Kiểm tra hộp lệch, sai lớp, trùng lặp và vật thể bị bỏ sót. |
| Instance segmentation | Mỗi đối tượng có nhãn lớp, `instance_id` và polygon/mask riêng. | Ảnh `kitchen`: cần kiểm tra vùng biên giữa người và vật dụng để xác định có bị che hoặc phân đoạn sai không. | Vẽ mask bám sát đường biên và tách riêng từng instance. | Kiểm tra mask có lấn sang nền, thiếu vùng vật thể hoặc gộp nhầm đối tượng không. |


## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu và hình ảnh công khai được cung cấp trong bài lab, không tải lên Google Colab hoặc GitHub bất kỳ dữ liệu cá nhân hay thông tin nội bộ nào khi chưa được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi bài lab, tôi sẽ dừng xử lý ngay, không tải lên hoặc chia sẻ dữ liệu đó và báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn xử lý phù hợp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS