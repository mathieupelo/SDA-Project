SELECT s.ticker, sp.close_price
FROM sda.stock_price sp
JOIN sda.stock s ON sp.stock_id = s.id
WHERE sp.date = '2024-09-16';
