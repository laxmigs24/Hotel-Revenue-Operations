
-- ================================================================
-- HOTEL REVENUE OPERATIONS — SQL QUERY LIBRARY
-- Author  : Laxmi Gupte
-- Dataset : Hotel Booking Demand (119,390 bookings, 2015–2017)
-- Table   : bookings (flat — all columns in one table)
-- ================================================================
-- HOW TO USE
-- Select any single query block and press Cmd+Return to run it.
-- ================================================================
 
 
-- ================================================================
-- STEP 0 — CREATE & POPULATE THE TABLE
-- Run this section first (only once) to set up the database.
-- ================================================================
 
CREATE TABLE IF NOT EXISTS bookings (
    booking_id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel                          TEXT,
    is_canceled                    INTEGER,   -- 1 = canceled, 0 = completed
    lead_time                      INTEGER,   -- days between booking and arrival
    arrival_year                   INTEGER,
    arrival_month                  TEXT,
    arrival_day                    INTEGER,
    weekend_nights                 INTEGER,
    week_nights                    INTEGER,
    adults                         INTEGER,
    children                       INTEGER,
    babies                         INTEGER,
    meal                           TEXT,
    country                        TEXT,
    market_segment                 TEXT,
    distribution_channel           TEXT,
    is_repeated_guest              INTEGER,
    previous_cancellations         INTEGER,
    previous_bookings_not_canceled INTEGER,
    reserved_room_type             TEXT,
    assigned_room_type             TEXT,
    booking_changes                INTEGER,
    deposit_type                   TEXT,
    days_in_waiting_list           INTEGER,
    customer_type                  TEXT,
    adr                            REAL,      -- average daily rate (€)
    required_car_parking_spaces    INTEGER,
    total_special_requests         INTEGER,
    reservation_status             TEXT
);
-- ================================================================
-- QUERY 1 — How many bookings do we have in total?
-- ================================================================
-- Shows the full size of the dataset split by outcome.
 
SELECT
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS total_canceled,
    COUNT(*) - SUM(is_canceled)                   AS total_completed,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings;
 
 
-- ================================================================
-- QUERY 2 — How many bookings does each hotel have?
-- ================================================================
-- Basic count split by hotel type.
 
SELECT
    hotel,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    COUNT(*) - SUM(is_canceled)                   AS completed
FROM bookings
GROUP BY hotel
ORDER BY total_bookings DESC;
 
 
-- ================================================================
-- QUERY 3 — Which year had the most bookings?
-- ================================================================
-- Useful for understanding data coverage across time.
 
SELECT
    arrival_year,
    COUNT(*)         AS total_bookings,
    SUM(is_canceled) AS canceled
FROM bookings
GROUP BY arrival_year
ORDER BY arrival_year;
 
 
-- ================================================================
-- QUERY 4 — What is the cancellation rate by market segment?
-- ================================================================
-- Shows which booking channels carry the most cancellation risk.
 
SELECT
    market_segment,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY market_segment
ORDER BY cancellation_rate_pct DESC;
 
 
-- ================================================================
-- QUERY 5 — Does deposit type affect cancellation?
-- ================================================================
-- Tests the business hypothesis that deposit behaviour predicts risk.
 
SELECT
    deposit_type,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY deposit_type
ORDER BY cancellation_rate_pct DESC;
 
 
-- ================================================================
-- QUERY 6 — What is the average daily rate (ADR) per hotel?
-- ================================================================
-- Compares pricing between City Hotel and Resort Hotel.
 
SELECT
    hotel,
    ROUND(AVG(adr), 2)  AS avg_adr,
    ROUND(MIN(adr), 2)  AS min_adr,
    ROUND(MAX(adr), 2)  AS max_adr
FROM bookings
WHERE adr > 0           -- exclude zero-rate / complimentary bookings
GROUP BY hotel;
 
 
-- ================================================================
-- QUERY 7 — Which months have the highest cancellation rate?
-- ================================================================
-- Reveals seasonal cancellation patterns across all years.
 
SELECT
    arrival_month,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY arrival_month
ORDER BY cancellation_rate_pct DESC;
 
 
-- ================================================================
-- QUERY 8 — Do guests with special requests cancel less often?
-- ================================================================
-- Tests whether engagement (special requests) signals commitment.
 
SELECT
    CASE
        WHEN total_special_requests = 0 THEN 'No special requests'
        WHEN total_special_requests = 1 THEN '1 special request'
        ELSE                                 '2 or more requests'
    END                                           AS request_group,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY request_group
ORDER BY cancellation_rate_pct DESC;
 
 
-- ================================================================
-- QUERY 9 — What is the average lead time for canceled
--           vs completed bookings?
-- ================================================================
-- Checks whether bookings made far in advance cancel more often.
 
SELECT
    CASE is_canceled
        WHEN 1 THEN 'Canceled'
        ELSE        'Completed'
    END                          AS outcome,
    COUNT(*)                     AS bookings,
    ROUND(AVG(lead_time), 1)     AS avg_lead_time_days,
    MIN(lead_time)               AS min_lead_time,
    MAX(lead_time)               AS max_lead_time
FROM bookings
GROUP BY is_canceled;
 
 
-- ================================================================
-- QUERY 10 — What is the estimated revenue lost to cancellations?
-- ================================================================
-- Calculates revenue at risk using ADR × total nights stayed.
 
SELECT
    hotel,
    COUNT(*)                                                         AS canceled_bookings,
    ROUND(SUM((weekend_nights + week_nights) * adr), 0)             AS total_revenue_at_risk,
    ROUND(AVG((weekend_nights + week_nights) * adr), 2)             AS avg_revenue_per_canceled_booking
FROM bookings
WHERE is_canceled = 1
  AND adr > 0
GROUP BY hotel
ORDER BY total_revenue_at_risk DESC;
 
 
-- ================================================================
-- QUERY 11 — Which customer types cancel the most?
-- ================================================================
 
SELECT
    customer_type,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY customer_type
ORDER BY cancellation_rate_pct DESC;
 
 
-- ================================================================
-- QUERY 12 — How does lead time relate to cancellation risk?
-- ================================================================
-- Groups lead time into buckets to show the pattern clearly.
 
SELECT
    CASE
        WHEN lead_time BETWEEN 0  AND 30  THEN '0-30 days'
        WHEN lead_time BETWEEN 31 AND 90  THEN '31-90 days'
        WHEN lead_time BETWEEN 91 AND 180 THEN '91-180 days'
        WHEN lead_time > 180              THEN '180+ days'
    END                                           AS lead_time_bucket,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY lead_time_bucket
ORDER BY cancellation_rate_pct DESC;
 
 
-- ================================================================
-- QUERY 13 — Do repeat guests cancel less than new guests?
-- ================================================================
 
SELECT
    CASE is_repeated_guest
        WHEN 1 THEN 'Repeat guest'
        ELSE        'New guest'
    END                                           AS guest_type,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct,
    ROUND(AVG(adr), 2)                            AS avg_adr
FROM bookings
GROUP BY is_repeated_guest;
 
 
-- ================================================================
-- QUERY 14 — What are the top 10 countries by number of bookings?
-- ================================================================
 
SELECT
    country,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
WHERE country IS NOT NULL
GROUP BY country
ORDER BY total_bookings DESC
LIMIT 10;
 
 
-- ================================================================
-- QUERY 15 — Which room type has the most upgrades?
-- ================================================================
-- An upgrade = reserved room type differs from assigned room type.
 
SELECT
    reserved_room_type,
    assigned_room_type,
    COUNT(*)   AS total_bookings,
    SUM(CASE WHEN reserved_room_type != assigned_room_type THEN 1 ELSE 0 END) AS upgrades,
    ROUND(
        100.0 * SUM(CASE WHEN reserved_room_type != assigned_room_type THEN 1 ELSE 0 END)
              / COUNT(*), 1
    )          AS upgrade_rate_pct
FROM bookings
GROUP BY reserved_room_type, assigned_room_type
ORDER BY upgrades DESC
LIMIT 10;
 
 
-- ================================================================
-- QUERY 16 — What is the average stay length per hotel?
-- ================================================================
 
SELECT
    hotel,
    ROUND(AVG(weekend_nights + week_nights), 1) AS avg_total_nights,
    ROUND(AVG(weekend_nights), 1)               AS avg_weekend_nights,
    ROUND(AVG(week_nights), 1)                  AS avg_week_nights
FROM bookings
WHERE (weekend_nights + week_nights) > 0
GROUP BY hotel;
 
 
-- ================================================================
-- QUERY 17 — How many bookings came through each
--            distribution channel?
-- ================================================================
 
SELECT
    distribution_channel,
    COUNT(*)                                      AS total_bookings,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct_of_total,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct
FROM bookings
GROUP BY distribution_channel
ORDER BY total_bookings DESC;
 
 
-- ================================================================
-- QUERY 18 — What meal plan do most guests choose,
--            and does it affect cancellation?
-- ================================================================
 
SELECT
    meal,
    COUNT(*)                                      AS total_bookings,
    SUM(is_canceled)                              AS canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1) AS cancellation_rate_pct,
    ROUND(AVG(adr), 2)                            AS avg_adr
FROM bookings
GROUP BY meal
ORDER BY total_bookings DESC;
 
 
-- ================================================================
-- QUERY 19 — Which bookings are highest risk?
-- ================================================================
-- Combines multiple risk signals into a simple score.
-- Score: 1 point per risk factor, max 4.
 
SELECT
    booking_id,
    hotel,
    market_segment,
    deposit_type,
    lead_time,
    adr,
    is_canceled,
    -- Add 1 point for each risk factor present
    (CASE WHEN lead_time > 180              THEN 1 ELSE 0 END
   + CASE WHEN deposit_type = 'Non Refund'  THEN 1 ELSE 0 END
   + CASE WHEN market_segment = 'Online TA' THEN 1 ELSE 0 END
   + CASE WHEN previous_cancellations > 0   THEN 1 ELSE 0 END) AS risk_score
FROM bookings
ORDER BY risk_score DESC, adr DESC
LIMIT 20;
 
 
-- ================================================================
-- QUERY 20 — Full summary: key metrics in one view
-- ================================================================
-- A single report a manager could read at a glance.
 
SELECT
    hotel,
    COUNT(*)                                                          AS total_bookings,
    SUM(is_canceled)                                                  AS total_canceled,
    ROUND(100.0 * SUM(is_canceled) / COUNT(*), 1)                     AS cancellation_rate_pct,
    ROUND(AVG(adr), 2)                                                AS avg_daily_rate,
    ROUND(AVG(lead_time), 0)                                          AS avg_lead_time_days,
    ROUND(AVG(weekend_nights + week_nights), 1)                       AS avg_stay_length_nights,
    ROUND(SUM(CASE WHEN is_canceled = 1 THEN (weekend_nights + week_nights) * adr ELSE 0 END), 0)
                                                                      AS total_revenue_at_risk
FROM bookings
GROUP BY hotel;