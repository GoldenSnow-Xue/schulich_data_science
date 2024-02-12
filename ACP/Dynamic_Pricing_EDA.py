#!/usr/bin/env python
# coding: utf-8

# # **Dynamic Pricing**

# ### **1. Dataset**
# 

# In[2]:


pip install matplotlib


# In[3]:


pip install seaborn


# ##### 1.1 Import Libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# ##### 1.2 Load Dataset

# In[6]:


df = pd.read_csv('/Users/frieda/Desktop/New Folder With Items/pmg-more-than-50-products.csv')
df


# In[ ]:


df.info()


# In[ ]:


df.describe(include='all').transpose()


# ### **2. Exploratory Data Analysis (EDA)**

# ##### 2.1 Data Cleaning

# ##### 2.1.1 Deal with Duplicates

# In[7]:


duplicates_counts = df.duplicated().sum()
print(duplicates_counts)


# This dataset has 3,761,425 duplicates which can drop it.

# In[8]:


df = df.drop_duplicates()


# In[ ]:


df.info()


# The original dataset has 34,023,438 entries, after dropping duplicates, the new dataset has 30,262,013 entries.

# ##### 2.1.2 Deal with Missing Values

# In[9]:


null_counts = df.isnull().sum()
print(null_counts)


# In[10]:


drop_maximum = 30262013 * 0.1 
print(f'drop_maximum: {drop_maximum}')


# The maximum amount that can drop is 3,026,201.
# 
# brand_name and standard_size_unit have null values which is more than 10% of the total data (drop_maximum), need to fill in.
# 
# standard_size_value, postal and province have null values which is less than the 10% of the total data (drop_maximum), simply drop them.

# In[11]:


df = df.dropna(subset=['standard_size_value', 'postal', 'province'])


# In[ ]:


null_counts = df.isnull().sum()
print(null_counts)


# After dropping the rows with the nulls values in those three columns, there are still a large of null values in 'brand_name' and "standard_size_unit'.
# 
# Try to find the relationship between columns to fill in the null values in these two columns.

# In[ ]:


product_id_grouped = df.groupby('product_id')['brand_name'].nunique()
non_unique_mappings = product_id_grouped[product_id_grouped > 1]
print(product_id_grouped)
print(non_unique_mappings)


# Each 'product_id' is consistently associated with a single brand, or no brand at all because the brand_name is missing.

# In[ ]:


product_matching_group_grouped = df.groupby('product_matching_group_id')['brand_name'].nunique()
non_unique_mappings = product_matching_group_grouped[product_matching_group_grouped > 1]
print(product_matching_group_grouped)
print(non_unique_mappings)


# There is more than one unique brand name for a single 'product_matching_group_id'.
# 
# Step 1: Find out the mode of the 'brand_name' to fill in the null values if have 'brand_name' record in the same 'product_matching_group_id'.
# 
# Step 2: Fill in the null values with 'Unknown' for those don't have any records in the same 'product_matching_group_id'.

# Before we fill in, make a copy of the original dataset.

# In[12]:


original_df = df.copy()


# In[ ]:


most_common_brand = df.groupby('product_matching_group_id')['brand_name'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.random.choice(x.dropna().unique()) if not x.dropna().empty else np.nan
)

brand_mapping = most_common_brand.to_dict()

df.loc[df['brand_name'].isnull(), 'brand_name'] = df['product_matching_group_id'].map(brand_mapping)

print(df['brand_name'].isnull().sum())


# Before fill in the null_values with the mode records in the 'brand_name', there are 6,240,795 null values, now only 641,723 null values left.
# 
# Fill these with 'Unknown' values.

# In[13]:


df['brand_name'].fillna('Unknown', inplace=True)


# For column 'standard_size_unit', fill in the null values in the same way.

# First, try to find out the relationship with other columns.

# In[ ]:


product_id_grouped_1 = df.groupby('product_id')['standard_size_unit'].nunique()
non_unique_mappings = product_id_grouped_1[product_id_grouped_1 > 1]
print(product_id_grouped_1)
print(non_unique_mappings)


# In[ ]:


product_matching_group_grouped_1 = df.groupby('product_matching_group_id')['standard_size_unit'].nunique()
non_unique_mappings = product_matching_group_grouped_1[product_matching_group_grouped_1 > 1]
print(product_matching_group_grouped_1)
print(non_unique_mappings)


# Before we fill in, make a copy of the original dataset.

# In[14]:


original_df_1 = df.copy()


# Find out the mode of the 'standard_size_unit' to fill in the null values if have 'standard_size_unit' record in the same 'product_matching_group_id'.

# In[ ]:


most_common_standard_size_unit = df.groupby('product_matching_group_id')['standard_size_unit'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.random.choice(x.dropna().unique()) if not x.dropna().empty else np.nan
)

standard_size_unit_mapping = most_common_standard_size_unit .to_dict()

df.loc[df['standard_size_unit'].isnull(), 'standard_size_unit'] = df['product_matching_group_id'].map(standard_size_unit_mapping)

print(df['standard_size_unit'].isnull().sum())


# In[ ]:


null_counts = df.isnull().sum()
print(null_counts)


# Now only 4848 null values left in the column 'standard_size_unit', simply drop it.

# In[ ]:


df = df.dropna(subset=['standard_size_unit'])


# In[ ]:


total_null_counts = df.isnull().sum().sum()
print(total_null_counts)


# Now there are no null values in this dataset.

# ##### 2.1.3 Deal with Inappropriate Range

# In[ ]:


df.describe(include="all").transpose()


# The column 'current_price' has the minimum value which is less than zero, it doesn't make sense, need to keep the dataset that the price is more than zero.

# In[ ]:


counts_current_price = (df['current_price'] <= 0).sum()
print(counts_current_price)


# In[15]:


df = df[df['current_price'] > 0]


# In[ ]:


df


# In[ ]:


df.info()


# ##### 2.1.4 Deal with Incorrect DataType

# Convert the 'collected_date' column from object datatype to datetime datatype.

# In[16]:


df = df.copy()


# In[17]:


df['collected_date'] = pd.to_datetime(df['collected_date'])


# ##### 2.1.5 Create Meaningful Columns for Further Analysis

# Create Year, Month and Week Columns

# In[18]:


df['Year'] = df['collected_date'].dt.year
df['Month'] = df['collected_date'].dt.month
df['Week'] = df['collected_date'].dt.isocalendar().week


# In[ ]:


df.head(2000000)


# In[ ]:


df.nunique()


# ##### 2.2 Data Visualization

# ##### 2.2.1 Numerical Variables

# In[ ]:


df.hist(figsize=(20,15))
plt.show()


# **Conclusion:**
# 
# **collected_date:**
#  Data collection has been increasing over time from 2020 to 2024, with some fluctuations.
# 
# **is_onsale:** 
# This histogram shows that there are significantly more instances where products are not on sale (0) compared to when they are on sale (1).
# 
# **Year:** 
# Data from the years 2020 to 2024 is displayed, with 2020 and 2023 having the most data points, suggesting more activity or data collected in these years.
# 
# **Month:**
# The data shows seasonal trends or monthly variations with peaks at certain times of the year, indicating perhaps seasonal sales or data collection trends.
# 
# **Week:**
# The data is spread across different weeks, with some weeks having more data points than others, which may indicate weekly sales cycles or reporting periods.
# 

# ##### 2.2.2 Boxplot to Check Outliers

# In[ ]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='current_price', y='category', data=df)
plt.title('Boxplot of Prices by Category')
plt.xlabel('Price')
plt.ylabel('Category')
plt.show()


# To ensure our analysis accurate, we're utilizing boxplots to visualize the distribution of prices across different product categories in our dataset. In this boxplot, we could find out different cateogries has different range of prices and outliers, so next step our purpose is to remove these outliers by category.

# ##### 2.2.3 Deal with Outliers for Column ['current_price'] by Category

# In[19]:


def remove_outliers(df, column, category_col):

    clean_df = pd.DataFrame()
    
    for category in df[category_col].unique():

        cat_df = df[df[category_col] == category]
        
        Q1 = cat_df[column].quantile(0.25)
        Q3 = cat_df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_cat_df = cat_df[(cat_df[column] >= lower_bound) & (cat_df[column] <= upper_bound)]
        
        clean_df = pd.concat([clean_df, filtered_cat_df], axis=0)
    
    return clean_df

cleaned_df = remove_outliers(df, 'current_price', 'category')

print(cleaned_df)


# This function is to identify the typical range of prices in each category by calculating the Interquartile Range (IQR), which is the range between the first quartile (25th percentile) and the third quartile (75th percentile).
# 
# Then define outliers as any price that lies more than 1.5 times the IQR below the first quartile or above the third quartile which is a standard practice for outlier detection.
# 
# Finally, Filtering out these outliers from our dataset to prevent them from skewing our analysis.

# In[ ]:


cleaned_df.info()


# In[ ]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='current_price', y='category', data=cleaned_df)
plt.title('Boxplot of Prices by Category')
plt.xlabel('Price')
plt.ylabel('Category')
plt.show()


# Now this is the distribution of prices across different product categories after the removal of outliers.
# 
# Pantry: Exhibits a wide range of prices with a number of outliers, indicating that while many pantry items are moderately priced, there are a few with significantly higher prices.
# 
# Produce: Shows a wide interquartile range with multiple outliers on the higher end, indicating a large variation in product prices within this category.
# 
# Meat/Seafood: Features one of the longest whiskers, especially on the higher end, which denotes a broad range of prices, and the presence of outliers suggests some items are priced much higher than others.
# 
# Health/Beauty: While the median price is moderate, this category has a few outliers, indicating that there are some items with prices that are much higher than the average.
# 
# Grocery: Has the lowest median price and a very narrow interquartile range, indicating that grocery items are generally low-priced with very little variation in price.

# In[ ]:


# Histogram and Kernel Density Plot
plt.figure(figsize=(12, 6))
sns.histplot(cleaned_df['current_price'], kde=True, color='blue', bins=50)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# This distribution uggests that most items are priced at the lower end of the scale, indicating a focus on affordable pricing. There are fewer items at higher prices, highlighting a limited presence in the premium market. 

# In[ ]:


#Kernel Density Estimation
plt.figure(figsize=(12, 6))
sns.kdeplot(cleaned_df['current_price'], color='purple', fill=True)
plt.title('Kernel Density Estimation of Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()


# The graph shows a high concentration of prices in the low range, peaking before 10, which indicates that most products are priced relatively low. The density drops off sharply as prices increase, with very few products priced over 20. This pattern emphasizes a product range that is largely budget-oriented, with limited offerings in the higher price segment.

# ##### 2.2.4 Distribution of Current Price

# **Distribution of Current Price per Company**

# In[ ]:


plt.figure(figsize=(15, 10))

sns.boxplot(x='company', 
            y='current_price', 
            data=cleaned_df
            )

plt.xticks(rotation=90)

plt.title('Current Price Distribution by Company')
plt.xlabel('Company')
plt.ylabel('Current Price')

plt.tight_layout()
plt.show()


# The boxplot reveals that while there is some variation in pricing among the companies, many have a similar range of prices. The presence of outliers across almost all companies suggests that each typically offers a few products priced well outside their general price range, likely to cater to niche markets or premium segments.

# **Distribution of Current Price per Province**

# In[ ]:


plt.figure(figsize=(15, 8))

sns.boxplot(x='province',
            y='current_price',
            data=cleaned_df
            )

plt.xticks(rotation=90)

plt.title('Current Price Distribution by Province')
plt.xlabel('Province')
plt.ylabel('Current Price')

plt.tight_layout()
plt.show()


# This boxplot suggests that, despite geographical differences, the central tendency of prices is comparable, but the range of prices reflects regional market differences.

# **Distribution of Current Price per Category**

# In[ ]:


plt.figure(figsize=(12, 8))

sns.boxplot(x='category',
            y='current_price',
            data=cleaned_df
            )

plt.xticks(rotation=90)

plt.title('Current Price Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Current Price (log scale)')

plt.tight_layout()
plt.show()


# The categories show varied price distributions, with some having a wider range indicating greater price diversity (such as meat/seafood and health/beauty), and others with a narrower range suggesting more consistent pricing (like grocery and organic). 
# 
# Notably, the meat/seafood and health/beauty categories have higher median prices and more high-priced outliers than other categories. In contrast, the grocery category has the lowest median price and fewer outliers, highlighting its position as the most budget-friendly option.
# 
# This information is valuable for understanding the pricing dynamics within each product category.

# In[ ]:


def month_to_season(month):
    if month in range(3, 5):
        return 'Spring'
    elif month in range(6, 8):
        return 'Summer'
    elif month in range(9, 11):
        return 'Fall'
    else:
        return 'Winter'

cleaned_df.loc[:, 'Season'] = cleaned_df['Month'].apply(month_to_season)

plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Season', data=cleaned_df, hue='is_onsale')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.xlabel('Season')
plt.ylabel('Count')
plt.title('Count of On Sale vs. Season')

plt.show()


# The chart indicates that, regardless of the season, there are always more items not on sale than on sale. However, the difference between the two categories is most pronounced in Winter, suggesting that it might be an off-peak season for sales or promotions. Conversely, the smaller gap in Fall could indicate more aggressive sales strategies during that season.

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2, 2, figsize=(20, 15))
                                            
sns.lineplot(x = 'Year', y = 'current_price', data=cleaned_df, ax=ax1)
sns.lineplot(x = 'Month', y = 'current_price', data=cleaned_df, ax=ax2)
sns.lineplot(x = 'Week', y = 'current_price', data=cleaned_df, ax=ax3)
sns.lineplot(x = 'Season', y = 'current_price', data=cleaned_df, ax=ax4)

ax1.set_title('Yearly Distribution')
ax2.set_title('Monthly Distribution')
ax3.set_title('Weekly Distribution')
ax4.set_title('Season Distribution')

plt.tight_layout()
plt.show()


# Yearly Distribution: This graph shows a general upward trend over the years from 2020 to 2024. It suggests that whatever is being measured (possibly prices, sales, etc.) is increasing over time.
# 
# Monthly Distribution: The data here fluctuates throughout the months, with noticeable dips and rises. There seems to be a pattern, possibly indicating seasonal effects on the data, with troughs and peaks at specific times of the year.
# 
# Weekly Distribution: This graph shows more volatility on a week-by-week basis, with several peaks and troughs indicating weekly variations in the data. There's no clear trend visible; instead, there are significant fluctuations from one week to another.
# 
# Season Distribution: Here, there's a pronounced dip in one season (labeled 'Summer'), followed by a sharp increase in the next season (labeled 'Fall'), and then it decreases again going into 'Spring'. This suggests that there is a significant seasonal impact on the data.

# ##### 2.3 Detail Visualization

# In[20]:


cleaned_df['province'] = cleaned_df['province'].replace({'British Columbia': 'BC', 'Saskatchewan': 'SK'})


# In[21]:


# Find all unique values in the 'company' column
unique_companies = cleaned_df['company'].unique()

# Print unique values
print("All unique companies:")
print(unique_companies)


# We've chosen Save-on-Foods as our example for the Exploratory Data Analysis (EDA):

# In[22]:


# Filter out the data where 'company' column equals 'Save-on-Foods'
save_on_foods_data = cleaned_df[cleaned_df['company'] == 'Save-on-Foods']

# 1. Descriptive statistics
print("\nDescriptive statistics for Save-on-Foods data:")
print(save_on_foods_data.describe())


# In[23]:


# Find all unique values in the 'company' column
unique_store = save_on_foods_data['store'].unique()

# Print unique values
print("All unique stores:")
print(unique_store)


# In[24]:


# Find all unique values in the 'company' column
unique_regular_price = save_on_foods_data['regular_price'].unique()

# Print unique values
print("All unique regular_price:")
print(unique_regular_price)


# In[25]:


save_on_foods_data_unique = save_on_foods_data.drop_duplicates(subset=['store_id'])

# 2. Visualization

sns.countplot(data=save_on_foods_data_unique, y='province', palette='viridis')
plt.xlabel('Number of Stores')
plt.ylabel('Province')
plt.title('Number of Unique Save-on-Foods Stores by Province')
plt.show()


# In[26]:


# Set the figure size
plt.figure(figsize=(10, 15))

# Create the count plot
sns.countplot(data=save_on_foods_data_unique, y='city', palette='viridis')

# Add labels and title
plt.xlabel('Number of Stores')
plt.ylabel('City')
plt.title('Number of Save-on-Foods Stores by City')

# Show the plot
plt.show()


# Based on the analysis, it is evident that Save-on-Foods has the highest number of stores in the province of British Columbia (BC). However, upon closer examination at the city level, it is observed that the city of Edmonton has the highest number of stores, despite being located in the province of Alberta (AB).
# 
# This discrepancy suggests several potential factors for further analysis:
# 1. **Market Demand**: Edmonton, being the capital city of Alberta, might have a higher population density or stronger market demand for Save-on-Foods stores compared to other cities in BC.
#   
# 2. **Competition**: The level of competition in the retail market could vary between provinces and cities. Factors such as the presence of competitors, market saturation, and consumer preferences may influence the distribution of Save-on-Foods stores.
# 
# 3. **Expansion Strategy**: Save-on-Foods may have adopted a strategic expansion plan that prioritizes certain regions or cities over others. Factors such as infrastructure development, economic growth, and demographic trends may influence the selection of locations for new stores.
# 
# 4. **Local Policies and Regulations**: Local policies, regulations, and zoning laws can also impact the establishment of retail outlets. Differences in municipal regulations between provinces may affect the ease of setting up stores in different cities.
# 
# 5. **Consumer Behavior**: Variations in consumer behavior, lifestyle preferences, and shopping habits between provinces and cities could influence the demand for Save-on-Foods stores. Factors such as income levels, cultural diversity, and urbanization rates may play a role.
# 
# In conclusion, while Save-on-Foods demonstrates a strong presence in the province of British Columbia overall, the dominance of Edmonton in terms of the number of stores within the province of Alberta highlights the importance of considering local market dynamics and strategic factors in retail expansion decisions. Further analysis incorporating demographic, economic, and competitive factors may provide deeper insights into the observed patterns.

# In[32]:


# Group by province and is_onsale, then count the occurrences
data_counts = save_on_foods_data.groupby(['province', 'is_onsale']).size().unstack()

# Plot with different colors for 'not on sale' and 'on sale'
data_counts.plot(kind='bar', stacked=True, figsize=(10, 5), color=['#FFA500', '#4169E1'])

plt.xlabel('Province')
plt.ylabel('Number of Stores')
plt.title('Save-on-Foods Stores Status by Province')
plt.legend(title='Status', labels=['Not on sale', 'On sale'])
plt.show()


# In[33]:


# Filter data for province 'AB'
province_AB_data = save_on_foods_data[save_on_foods_data['province'] == 'AB']

# Group by category and count the occurrences of 'is_onsale' for each category
category_sale_counts = province_AB_data.groupby('category')['is_onsale'].value_counts().unstack().fillna(0)

# Plotting the bar chart
category_sale_counts.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.title('Number of Products On Sale and Not On Sale by Category in Province AB')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='On Sale', labels=['Not On Sale', 'On Sale'])  # Add legend labels
plt.show()


# Based on the analysis of the sales data, it's apparent that across all provinces, there are slightly more products not on sale compared to those on sale in Save-on-Foods stores. Furthermore, upon closer examination of the province of Alberta (AB), it is observed that the majority of products on sale fall under the 'Pantry' category, and the store inventory predominantly consists of pantry items.
# 
# This observation leads to several conclusions:
# 
# 1. **Sales Strategy**: Save-on-Foods may employ a sales strategy that prioritizes offering discounts on specific categories of products, such as pantry items, to attract customers and drive sales. 
# 
# 2. **Consumer Preferences**: The prevalence of pantry items on sale suggests that these products may be in higher demand among consumers, prompting the retailer to offer discounts to incentivize purchases.
# 
# 3. **Inventory Composition**: The focus on pantry items in the sales promotions could reflect the composition of Save-on-Foods' inventory, indicating a strategic emphasis on stocking essential household goods.
# 
# 4. **Profit Margins**: The higher proportion of non-sale items could indicate that non-discounted products contribute more significantly to the retailer's profit margins, leading to a deliberate balance between sales promotions and regular pricing strategies.
# 
# 5. **Competitive Landscape**: The sales patterns observed in Alberta may also be influenced by the competitive landscape, consumer behavior, and local market dynamics specific to the province.
# 
# In summary, the analysis highlights the prevalence of pantry items on sale in Save-on-Foods stores across Alberta, suggesting a targeted sales approach aimed at meeting consumer demand while optimizing profitability. Understanding these sales trends can inform strategic decisions related to inventory management, pricing strategies, and customer engagement initiatives.

# In[ ]:


# Group by Month and count the occurrences of 'is_onsale' for each month
month_sale_counts = save_on_foods_data.groupby('Month')['is_onsale'].value_counts().unstack().fillna(0)

# Plotting the bar chart
ax = month_sale_counts.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.xlabel('Month')
plt.ylabel('Number of Products')
plt.title('Number of Products On Sale and Not On Sale by Month')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='On Sale', labels=['Not On Sale', 'On Sale'])  


# In[36]:


# Count of transactions by Province
num_products_by_province = save_on_foods_data.groupby('province').size()


# Displaying the results
print("Count of num_products by Province:")
print(num_products_by_province)


# In[ ]:


units_by_category = save_on_foods_data.groupby('category')['is_onsale'].aggregate(['sum','mean','median','count'])
units_by_category.head()


# Based on the analysis conducted:
# 
# 1. **Seasonal Variation in Products on Sale**: The bar chart depicting the number of products on sale and not on sale by month reveals a seasonal pattern. Specifically, there is a higher number of products during the winter season, a significant decrease in products during the spring, and another peak during the summer.
# 
# 2. **Num Counts by Province**: The transaction counts by province indicate that British Columbia (BC) has the highest number of transactions, followed by Alberta (AB). Manitoba (MB), Saskatchewan (SK), and Yukon (YT) have comparatively lower transaction counts.
# 
# 3. **Units Sold by Category**: Upon further analysis, it is observed that the 'dairy' and 'drinks' categories have the highest total units sold, followed by 'bakery'. However, the 'deli' category has relatively lower units sold, suggesting potentially lower demand or availability for deli products.
# 

# 
# Next steps:
# 
# Exploratory Data Analysis (EDA): Concentrate on a single city or province for all companies.
# 
# Testing: Assess the Impact of Time (Month, Year) on Sales: Utilize the collected date (collected_date) information to investigate if there are significant variations on sale items over time (e.g., across different months or years). This can be accomplished by employing ANOVA or regression analysis, treating time as either a categorical or continuous predictor variable.
