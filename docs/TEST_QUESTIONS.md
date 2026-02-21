# Bộ câu hỏi test hệ thống AI Agent

Dùng để kiểm thử routing (search / order / api / chat), chất lượng trả lời và xử lý biên.

---

## 1. Tìm kiếm sản phẩm (Intent: **search**)

### 1.1 Theo thương hiệu
- Tôi muốn xem điện thoại Samsung.
- Có iPhone nào đang giảm không?
- Xiaomi giá rẻ có mẫu nào?
- Tìm máy OnePlus dưới 50 triệu.
- Oppo hoặc Vivo pin trâu.

### 1.2 Theo giá
- Điện thoại dưới 10 triệu.
- Mua máy tầm 15–20 triệu.
- Tìm laptop trên 25 triệu.
- Điện thoại giá từ 8 đến 12 triệu.

### 1.3 Theo nhu cầu / thông số
- Điện thoại chơi game, ram 8GB.
- Máy nào camera đẹp?
- Pin trâu, dùng cả ngày.
- Màn hình lớn, xem phim.
- So sánh Samsung S24 và iPhone 15.

### 1.4 Đa danh mục (Laptop, Tablet, Phụ kiện)
- Laptop văn phòng dưới 20 triệu.
- Tablet cho con học online.
- Tai nghe chụp tai dưới 2 triệu.
- Sạc nhanh 65W cho laptop.

### 1.5 Câu dài / phức tạp
- Tôi cần mua điện thoại chơi game, pin trâu, camera tốt, giá khoảng 12–15 triệu, ưu tiên Samsung hoặc Xiaomi.
- Gợi ý giúp tôi 2–3 máy so sánh: một máy chụp ảnh đẹp, một máy chơi game, tầm 15 triệu.

---

## 2. Đơn hàng & thanh toán (Intent: **order**)

### 2.1 Trạng thái đơn
- Đơn hàng #1234 của tôi đến đâu rồi?
- Cho tôi tra cứu đơn 5678.
- Đơn hàng 9999 trạng thái thế nào?
- Giao hàng đơn 12345 khi nào tới?

### 2.2 Hủy / đổi / trả
- Tôi muốn hủy đơn 1234.
- Đổi trả đơn 5678 được không?
- Trả hàng đơn 9999 thì làm sao?

### 2.3 Thanh toán
- Thanh toán đơn 1234 bằng thẻ.
- Hóa đơn đơn 5678 gửi cho tôi.
- Đơn 9999 tổng tiền bao nhiêu?

### 2.4 Không có số đơn
- Đơn hàng của tôi ở đâu? *(kỳ vọng: hỏi lại số đơn)*
- Tra cứu giúp tôi đơn hàng. *(kỳ vọng: hỏi lại số đơn)*

---

## 3. Hội thoại chung (Intent: **chat**)

### 3.1 Chào hỏi
- Xin chào.
- Chào bạn, bạn có thể giúp tôi không?
- Hi, tôi cần tư vấn.

### 3.2 Cảm ơn & kết thúc
- Cảm ơn bạn.
- Thanks, bye.
- Được rồi, tạm biệt.

### 3.3 Hỏi hỗ trợ chung
- Bạn làm được những gì?
- Giúp tôi với.
- Hỗ trợ khách hàng ở đâu?
- Chính sách bảo hành thế nào? *(có thể chat hoặc api nếu có endpoint bảo hành)*

### 3.4 Câu không rõ ràng
- Ok.
- ??? 
- Không biết nữa.
- Bạn nghĩ sao?

---

## 4. API / Dịch vụ (Intent: **api**)

- Tích hợp API đơn hàng như thế nào?
- Webhook cho đơn hàng có không?
- Kết nối dịch vụ thanh toán?
- Service tra cứu bảo hành?

---

## 5. Biên & an toàn

### 5.1 Câu rất ngắn / rỗng
- *(chuỗi rỗng)*
- "   " *(chỉ khoảng trắng)*
- 1
- .

### 5.2 Câu rất dài
- Một đoạn văn > 500 từ lặp lại “điện thoại Samsung” nhiều lần.

### 5.3 Nhầm intent (search vs chat)
- “Điện thoại” *(có thể search hoặc chat tùy rule)*
- “Có Samsung không?” *(search)*
- “Samsung là gì?” *(chat)*

### 5.4 Đa ý trong một câu
- Xin chào, tôi muốn tra đơn 1234 và tìm thêm điện thoại dưới 10 triệu. *(kỳ vọng: xử lý 1 trong 2 hoặc hỏi làm rõ)*

### 5.5 Ký tự đặc biệt
- Tìm điện thoại @#$%.
- <script>alert(1)</script>.
- " OR 1=1 --

---

## 6. Test với user_id / session_id

- Gửi cùng câu search 2 lần với cùng `user_id` + `session_id`: lần 2 có thể cache (nếu cache bật).
- Gửi “đơn 1234” với `user_id` có giá trị vs không có: kiểm tra auth required cho order.
- Gửi “xin chào” rồi “tìm Samsung” trong cùng `session_id`: kiểm tra session/context (nếu đã implement).

---

## 7. Ghi chú khi chạy test

| Kiểm tra | Ghi chú |
|----------|--------|
| **Intent** | So sánh intent trả về với intent kỳ vọng trong bảng trên. |
| **Search** | RAG bật: có kết quả sản phẩm; RAG tắt: fallback LLM, không crash. |
| **Order** | `ENABLE_API_CALLS=true`: gọi Spring Boot; `false`: message “API đang tắt”. |
| **Chat** | Câu trả lời bằng tiếng Việt, không bịa thông tin sản phẩm. |
| **Latency** | Ghi nhận thời gian phản hồi (search thường chậm hơn chat). |
| **Cache** | Cùng query 2 lần: lần 2 nhanh hơn và metadata có `cached: true` (nếu có). |

---

## 8. Chạy nhanh bằng curl

```bash
# Health
curl -s http://localhost:8000/health | jq .

# Search
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"Tìm điện thoại Samsung dưới 20 triệu","user_id":"test1"}' | jq .

# Order (cần user_id để không bị auth_required)
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"Đơn hàng #1234 ở đâu?","user_id":"test1"}' | jq .

# Chat
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"Xin chào","user_id":"test1"}' | jq .
```

---

*Cập nhật thêm câu hỏi khi có tính năng mới hoặc rule mới.*
