DROP VIEW sda.v_stock_prices_2010;

CREATE OR REPLACE VIEW sda.v_stock_prices AS
SELECT 
    sp.date,
    s.name AS stock_name,
    s.ticker AS stock_ticker,
    sp.close_price
FROM 
    sda.stock_price sp
JOIN 
    sda.stock s ON sp.stock_id = s.id
WHERE 
    sp.close_price IS NOT NULL;
