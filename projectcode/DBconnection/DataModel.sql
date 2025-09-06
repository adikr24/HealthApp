drop table FoundationFoodInformation.food;

CREATE TABLE FoundationFoodInformation.food (
    fdc_id BIGINT PRIMARY KEY,
    data_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    food_category_id VARCHAR(10),
    publication_date DATE
);


drop table if exists FoundationFoodInformation.food_attribute_type;

CREATE TABLE FoundationFoodInformation.food_attribute_type (
    id_ BIGINT PRIMARY KEY,
    name_ VARCHAR(50) NOT NULL,
    description TEXT NOT NULL
);


drop table if exists FoundationFoodInformation.food_calorie_conversion_factor;

CREATE TABLE FoundationFoodInformation.food_calorie_conversion_factor (
    food_nutrient_conversion_factor_id BIGINT PRIMARY KEY,
    protein_value DECIMAL(8, 4),  -- Example: Stores values like 4.0000 (4 digits after decimal)
    fat_value DECIMAL(8, 4),      -- Adjust precision and scale as needed
    carbohydrate_value DECIMAL(8, 4)
);

drop table if exists FoundationFoodInformation.food_nutrient;

CREATE TABLE FoundationFoodInformation.food_nutrient (
    id INT PRIMARY KEY,
    fdc_id INT,
    nutrient_id INT,
    amount TEXT,
    data_points TEXT,
    derivation_id TEXT,
    min TEXT,   -- Store as text
    max TEXT,   -- Store as text
    median TEXT, -- Store as text
    footnote TEXT,
    min_year_acquired TEXT , -- Store as text
	FOREIGN KEY (fdc_id) REFERENCES FoundationFoodInformation.food(fdc_id)
);


--- the FDC_ID in FOODS were missing so there wasn't a 1:1 connection found ----
drop table if exists FoundationFoodInformation.food_nutrient_conversion_factor;

CREATE TABLE FoundationFoodInformation.food_nutrient_conversion_factor (
    id_ INT PRIMARY KEY,
    fdc_id INT
   --  FOREIGN KEY (fdc_id) REFERENCES FoundationFoodInformation.food(fdc_id)
    -- Assuming 'foods' is the table that 'fdc_id' refers to
);

select * from FoundationFoodInformation.food_nutrient_conversion_factor; 

CREATE TABLE foundationfoodinformation.food_attribute (
    id SERIAL PRIMARY KEY,
    fdc_id INT,
    seq_num TEXT,
    food_attribute_type_id TEXT,
    name VARCHAR(255),
    value TEXT
);


select * from foundationfoodinformation.food_attribute;

drop table if exists foundationfoodinformation.food_component;
CREATE TABLE foundationfoodinformation.food_component (
    id SERIAL PRIMARY KEY,
    fdc_id TEXT,
    name VARCHAR(255),
    pct_weight TEXT,
    is_refuse BOOLEAN,
    gram_weight TEXT,
    data_points TEXT,
    min_year_acquired TEXT
);



drop table if exists foundationfoodinformation.nutrient;
CREATE TABLE foundationfoodinformation.nutrient (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    unit_name VARCHAR(50),
    nutrient_nbr Text,
    rank Text
);



---------- Daily nutritional values, Macro nutrient ------------------

drop table if exists dailyval.macronutrients;
CREATE TABLE dailyval.macronutrients (
    Component VARCHAR(50) NOT NULL,
    Unit_of_Measure VARCHAR(50) NOT NULL,
    Adults_and_Children_4_years INT NULL,
    Infants_through_12_months INT NULL,
    Children_1_through_3_years INT NULL,
    Pregnant_and_Lactating_Women INT NULL,
    Unit_of_Measure_SI VARCHAR(10) NOT NULL,
    PRIMARY KEY (Component)
);


INSERT INTO dailyval.macronutrients 
(Component, Unit_of_Measure, Adults_and_Children_4_years, Infants_through_12_months, Children_1_through_3_years, Pregnant_and_Lactating_Women, Unit_of_Measure_SI)
VALUES
('Fat', 'Grams (g)', 78, 30, 39, 78, 'g'),
('Saturated fat', 'Grams (g)', 20, NULL, 10, 20, 'g'),
('Cholesterol', 'Milligrams (mg)', 300, NULL, 300, 300, 'mg'),
('Total carbohydrates', 'Grams (g)', 275, 95, 150, 275, 'g'),
('Sodium', 'Milligrams (mg)', 2300, NULL, 1500, 2300, 'mg'),
('Dietary Fiber', 'Grams (g)', 28, NULL, 14, 28, 'g'),
('Protein', 'Grams (g)', 50, NULL, 13, NULL, 'g'),
('Added sugars', 'Grams (g)', 50, NULL, 25, 50, 'g');



CREATE TABLE dailyval.micronutrients (
    Nutrient VARCHAR(50) NOT NULL,
    Unit_of_Measure VARCHAR(25) NOT NULL,
    Adults_and_Children_4_years DECIMAL(8, 2) NULL,
    Infants_through_12_months DECIMAL(8, 2) NULL,
    Children_1_through_3_years DECIMAL(8, 2) NULL,
    Pregnant_and_Lactating_Women DECIMAL(8, 2) NULL,
    Unit_of_Measure_SI VARCHAR(10) NOT NULL,
    PRIMARY KEY (Nutrient)
);

INSERT INTO dailyval.micronutrients (Nutrient, Unit_of_Measure, Adults_and_Children_4_years, Infants_through_12_months, Children_1_through_3_years, Pregnant_and_Lactating_Women, Unit_of_Measure_SI)
VALUES
('Vitamin A', 'Micrograms RAE2 (mcg)', 900.00, 500.00, 300.00, 1300.00, 'mcg'),
('Vitamin C', 'Milligrams (mg)', 90.00, 50.00, 15.00, 120.00, 'mg'),
('Calcium', 'Milligrams (mg)', 1300.00, 260.00, 700.00, 1300.00, 'mg'),
('Iron', 'Milligrams (mg)', 18.00, 11.00, 7.00, 27.00, 'mg'),
('Vitamin D', 'Micrograms (mcg)3', 20.00, 10.00, 15.00, 15.00, 'mcg'),
('Vitamin E', 'Milligrams (mg)4', 15.00, 5.00, 6.00, 19.00, 'mg'),
('Vitamin K', 'Micrograms (mcg)', 120.00, 2.50, 30.00, 90.00, 'mcg'),
('Thiamin', 'Milligrams (mg)', 1.20, 0.30, 0.50, 1.40, 'mg'),
('Riboflavin', 'Milligrams (mg)', 1.30, 0.40, 0.50, 1.60, 'mg'),
('Niacin', 'Milligrams NE5 (mg)', 16.00, 4.00, 6.00, 18.00, 'mg'),
('Vitamin B6', 'Milligrams (mg)', 1.70, 0.30, 0.50, 2.00, 'mg'),
('Folate6', 'Micrograms DFE7 (mcg)', 400.00, 80.00, 150.00, 600.00, 'mcg'),
('Vitamin B12', 'Micrograms (mcg)', 2.40, 0.50, 0.90, 2.80, 'mcg'),
('Biotin', 'Micrograms (mcg)', 30.00, 6.00, 8.00, 35.00, 'mcg'),
('Pantothenic acid', 'Milligrams (mg)', 5.00, 1.80, 2.00, 7.00, 'mg'),
('Phosphorus', 'Milligrams (mg)', 1250.00, 275.00, 460.00, 1250.00, 'mg'),
('Iodine', 'Micrograms (mcg)', 150.00, 130.00, 90.00, 290.00, 'mcg'),
('Magnesium', 'Milligrams (mg)', 420.00, 75.00, 80.00, 400.00, 'mg'),
('Zinc', 'Milligrams (mg)', 11.00, 3.00, 3.00, 13.00, 'mg'),
('Selenium', 'Micrograms (mcg)', 55.00, 20.00, 20.00, 70.00, 'mcg'),
('Copper', 'Milligrams (mg)', 0.90, 0.20, 0.30, 1.30, 'mg'),
('Manganese', 'Milligrams (mg)', 2.30, 0.60, 1.20, 2.60, 'mg'),
('Chromium', 'Micrograms (mcg)', 35.00, 5.50, 11.00, 45.00, 'mcg'),
('Molybdenum', 'Micrograms (mcg)', 45.00, 3.00, 17.00, 50.00, 'mcg'),
('Chloride', 'Milligrams (mg)', 2300.00, 570.00, 1500.00, 2300.00, 'mg'),
('Potassium', 'Milligrams (mg)', 4700.00, 700.00, 3000.00, 5100.00, 'mg'),
('Choline', 'Milligrams (mg)', 550.00, 150.00, 200.00, 550.00, 'mg');





