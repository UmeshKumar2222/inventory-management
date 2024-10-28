# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import streamlit as st
import logging
import openai
import io
from mlxtend.frequent_patterns import apriori, association_rules



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Azure OpenAI API client setup
openai.api_key = "bd18995c51fa40e19e493df21c7ded81"
openai.api_base = "https://madhukar-kumar.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

def pareto(df, col, saleAmt):
    total = df[saleAmt].sum()
    sales_80 = total * 0.8
    df_pareto = df.groupby(col)[saleAmt].sum().sort_values(ascending=False).reset_index()
    df_pareto['cumSale'] = df_pareto[saleAmt].cumsum()
    df_80 = df_pareto[df_pareto['cumSale']<= sales_80]
    total_row = df[col].nunique()
    row80 = df_80[col].nunique()
    percnt = row80 / total_row *100
    return df_80,percnt

# Function to get completion from Azure OpenAI
def get_completion(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",  # Replace with your actual model deployment name in Azure
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        logger.info("Successfully retrieved completion from OpenAI.")
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {e}")
        st.error("Error connecting to the OpenAI API. Please check your network connection.")
        return ""

# Function to generate insights based on DataFrame or plot description
def generate_insight(description: str) -> str:
    prompt = f"Provide a summary or insight in bullet points based on the following data or graph description: {description}"
    return get_completion(prompt)

# Function to generate insights based on trend data
def generate_graph_insight(trend_data: pd.Series) -> str:
    try:
        # # Extract basic information from the trend data (min, max, average)
        # min_value = trend_data.min()
        # max_value = trend_data.max()
        # avg_value = trend_data.mean()

        # # Extract insights about trends
        # trend_description = ""
        # if trend_data.is_monotonic_increasing:
        #     trend_description = "The data shows a consistent increase over time."
        # elif trend_data.is_monotonic_decreasing:
        #     trend_description = "The data shows a consistent decrease over time."
        # else:
        #     trend_description = "The data shows fluctuations over time with peaks and valleys."

        # # Describe statistical information
        # description = (
        #     f"The trend data ranges from a minimum of {min_value:.2f} to a maximum of {max_value:.2f}, "
        #     f"with an average value of {avg_value:.2f}. {trend_description}"
        # )
        
        # # Generate insight based on extracted graph data
        prompt = f"Provide insights based on the following trend data for timeseries .explain the key points like seasonality and trend. explain in the form of easily understandable 5 bullet points so that non tech person can also understans: {trend_data}"
        return get_completion(prompt)
    except Exception as e:
        logger.error(f"Error generating graph insights: {e}")
        return "Error generating insights for the graph."

st.set_page_config(
    page_title="Inventory Management",
    layout='wide'
)


tab1, tab3 , tab2, tab4, tab5= st.tabs(["Inventory", "Purchase", "Sales", "Customer", "Product"])




with tab1:
    beg_inv = None
    end_inv = None
    st.title("Inventory Analysis")
    

    # Upload Dataset for Beginning Inventory
    if 'data1' not in st.session_state:
        if st.button("Upload Dataset for Beginning Inventory"):
            st.switch_page("data_upload.py")

    else:
        beg_inv = st.session_state.data1

    if 'data2' not in st.session_state:
        if st.button("Upload Dataset for End Inventory"):
            st.switch_page("data_upload.py")
    else:
        end_inv = st.session_state.data2
    if beg_inv is not None and end_inv is not None:
        try:
            Brand = st.selectbox("Select column which contains Brand of product", beg_inv.columns, index=None)
            description = st.selectbox("Select column which contains Description of product", beg_inv.columns, index=None)
            Qnt = st.selectbox("Select column which contains Quantity of product", beg_inv.columns, index=None)

            if all([Brand is not None, description is not None, Qnt is not None]):

                # Grouping by Brand and Description and summarize inventory for beginning of the year
                beg_summary = beg_inv.groupby([Brand, description])[Qnt].sum().sort_values(ascending=False)
                
                # Grouping by Brand and Description and summarize inventory for end of the year
                end_summary = end_inv.groupby([Brand, description])[Qnt].sum().sort_values(ascending=False)

                # Identifying top 5 products at the beginning and end of the year
                top_5_beg = beg_summary.head(5)
                top_5_end = end_summary.head(5)

                # Identifying bottom 5 products at the beginning and end of the year
                bottom_5_beg = beg_summary.tail(5)
                bottom_5_end = end_summary.tail(5)

                left, right = st.columns(2)
                left1, right1 = st.columns(2)
                with left:
                    st.write("Top 5 products at the beginning of the year:")
                    st.dataframe(top_5_beg)
                    df_description = top_5_beg
                    insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                    st.write("Insights for Top 5 products at the beginning of the year:")
                    st.write(insight)

                with left1:  
                    st.write("Bottom 5 products at the beginning of the year:")
                    st.dataframe(bottom_5_beg)
                    df_description = bottom_5_beg
                    insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                    st.write("Insights for Bottom 5 products at the beginning of the year:")
                    st.write(insight)

                with right:
                    st.write("Top 5 products at the end date:")
                    st.dataframe(top_5_end)
                    df_description = top_5_end
                    insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                    st.write("Insights for Top 5 products at the end date:")
                    st.write(insight)

                with right1:
                    st.write("Bottom 5 products at the end date:")
                    st.dataframe(bottom_5_end)
                    df_description = bottom_5_end  # Fixed the variable name
                    insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                    st.write("Insights for Bottom 5 products at the end date:")
                    st.write(insight)

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")
            st.error(f"An error occurred during analysis: {e}")

with tab2:
    st.title("Sales Analysis")
    sales = None
    if 'data4' not in st.session_state:
        if st.button("Upload Dataset for Sales"):
            st.switch_page("data_upload.py")
    else:
        sales = st.session_state.data4


    if sales is not None:
        try:
            Brand = st.selectbox("Select column which contains Brand of product", sales.columns, index=None)
            description = st.selectbox("Select column which contains Description of product", sales.columns, index=None)
            Qnt = st.selectbox("Select column which contains Quantity of product", sales.columns, index=None)
            s_date = st.selectbox("Select column which contains date of product sale", sales.columns, index=None)
            sale_amt = st.selectbox("Select column which contain sale amount", sales.columns, index=None)




            st.write("Dataset:")
            st.dataframe(sales.head())
            df_description = sales.describe().to_string()
            insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
            st.write("Insight for DataFrame:")
            st.write(insight)
            if all([Brand is not None, description is not None, Qnt is not None, s_date is not None, sale_amt is not None]):
                # Finding best-selling product
                best_selling_product = sales.groupby([Brand, description]).agg({Qnt: 'sum'}).sort_values(by=Qnt, ascending=False)
                st.write("Top 10 Selling Products:")    #, best_selling_product.head(10))
                df_description = best_selling_product.head(10)
                # Bar graph for top vendors by purchase cost
                fig1, ax = plt.subplots(figsize=(10, 6))
                # df_description.plot(kind='bar', color='coral', ax=ax)
                bars = df_description.plot(kind='bar', color='coral', ax=ax)
                for bar in bars.patches:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                            int(bar.get_height()), ha='center', va='bottom')
                plt.title('Top 10 Product by Sales Quantity')
                plt.ylabel('Sales Quantity')
                plt.xlabel('Product Description')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig1)

                insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                st.write("Insight for Best Selling Products:")
                st.write(insight)

                # Finding the slow-moving products
                slow_moving_products = sales.groupby([Brand, description]).agg({Qnt: 'sum'}).sort_values(by=Qnt, ascending=True).head(10)
                st.write("Slow-Moving Ten Products:")#, slow_moving_products.head(10))
                df_description = slow_moving_products
                # Bar graph for top vendors by purchase cost
                fig1, ax = plt.subplots(figsize=(10, 6))
                # df_description.plot(kind='bar', color='coral', ax=ax)
                bars = df_description.plot(kind='bar', color='coral', ax=ax)
                for bar in bars.patches:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                            int(bar.get_height()), ha='center', va='bottom')
                plt.title('Bottom 10 Product by Sales Quantity')
                plt.yticks(range(0, 31, 5))
                plt.ylabel('Sales Quantity')
                plt.xlabel('Product Description')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig1)
                insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                st.write("Insight for Slow-Moving Products:")
                st.write(insight)

                # Sales quantity trend
                st.header("Seasonality and Trend")
                sales[s_date] = pd.to_datetime(sales[s_date])
                sales_quantity_trend = sales.groupby(s_date)[Qnt].sum()
                fig, ax = plt.subplots(figsize=(12, 6))
                sales_quantity_trend.plot(ax=ax)
                ax.set_title('Sales Quantity Over Time')
                st.pyplot(fig)

                # Generate insight for the graph based on actual data
                graph_insight = generate_graph_insight(sales_quantity_trend)
                st.write("Insight for Graph:")
                st.write(graph_insight)

                total_sales = sales[sale_amt].sum()
                st.write("Total Sales: ",round(total_sales, 2))
                
                df_80, percnt = pareto(sales,description,sale_amt)
                st.header(f"80% of the total sales is done by {round(percnt, 2)}% product.")
                st.dataframe(df_80[[description,sale_amt]].head(10))

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")
            st.error(f"An error occurred during analysis: {e}")

with tab3:
    st.title("Purchase Analysis")
    purchases = None
    if 'data3' not in st.session_state:
        if st.button("Upload Dataset for Purchase Final"):
            st.switch_page("data_upload.py")
    else:
        purchases = st.session_state.data3
    if purchases is not None:
        try:
            # Column selections
            vendor = st.selectbox("Select column which contains Vendor of product", purchases.columns, index=None)
            purchase_price = st.selectbox("Select column which contains Purchase Price of product", purchases.columns, index=None)
            Qnt = st.selectbox("Select column which contains Quantity of product", purchases.columns, index=None)
            
            if all([vendor is not None, purchase_price is not None, Qnt is not None]):
                # Vendor Purchase Volume
                col1, col2 = st.columns(2)
                with col1:
                    vendor_purchase_volume = purchases.groupby(vendor).agg({Qnt: 'sum'}).sort_values(by=Qnt, ascending=False)
                    st.write("Top 10 Vendors by Purchase Volume:\n", vendor_purchase_volume.head(10))
                
                with col2:
                    vendor_purchase_cost = purchases.groupby(vendor).agg({purchase_price: 'sum'}).sort_values(by=purchase_price, ascending=False)
                    st.write("Top 10 Vendors by Purchase Cost:\n", vendor_purchase_cost.head(10))

                # Top Vendors Bar Graph
                reduced_purchases = purchases[[vendor, purchase_price]]
                top_vendors = reduced_purchases.groupby(vendor).sum()[purchase_price].nlargest(10)

                # Bar graph for top vendors by purchase cost
                fig1, ax = plt.subplots(figsize=(10, 6))
                top_vendors.plot(kind='bar', color='coral', ax=ax)
                plt.title('Top 10 Vendors by Purchase Cost')
                plt.ylabel('Purchase Cost')
                plt.xlabel('Vendor Name')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig1)

                # Generate insights
                insight = generate_insight(f"This is a summary of the following DataFrame:\n{top_vendors}")
                st.write("Insight for top vendors:")
                st.write(insight)

                # Pie chart for distribution of purchase costs among top vendors
                fig, ax = plt.subplots(figsize=(6, 4))
                top_vendors.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
                plt.title('Distribution of Purchase Costs Among Top Vendors')
                plt.ylabel('')  # to remove the default 'Purchase Price' label from the y-axis
                st.pyplot(fig)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            st.error(f"Error during analysis: {e}")

with tab4:
    st.title("Customer Analysis")
    df  = None
    if 'data4' not in st.session_state:
        if st.button("Upload Dataset"):
            st.switch_page("Data_Upload.py")
    else:
        df = st.session_state.data4 

    if df is not None:
        
        cust = st.selectbox("Select your customer ID column", df.columns, index=None)
        ord = st.selectbox("Select Invoice Date column", df.columns, index =None)
        ordNo = st.selectbox("Select Invoice Number column", df.columns, index =None)
        price = st.selectbox("Select Sale amount column", df.columns, index =None)
        
        if all([cust is not None, ord is not None, ordNo is not None, price is not None]):
            try:
                df[ord] = pd.to_datetime(df[ord])
                
                # Calculate last day
                last_day = df[ord].max() + dt.timedelta(days = 1)
                rfm_table = df.groupby(cust).agg({ord: lambda x: (last_day - x.max()).days,
                                                    ordNo: "nunique",
                                                    price: "sum"})

                rfm_table.rename(columns = {ord: "Recency",
                                            ordNo: "Frequency",
                                            price : "Monetary"}, inplace = True)

                r_labels = range(5, 0, -1)
                fm_labels = range(1, 6)

                rfm_table["R"] = pd.qcut(rfm_table["Recency"], 5, labels = r_labels)
                rfm_table["F"] = pd.qcut(rfm_table["Frequency"].rank(method = 'first'), 5, labels = fm_labels)
                rfm_table["M"] = pd.qcut(rfm_table["Monetary"], 5, labels = fm_labels)

                # st.dataframe(rfm_table.head())
                rfm_table["RFM_Segment"] = rfm_table["R"].astype(str) + rfm_table["F"].astype(str) + rfm_table["M"].astype(str)
                rfm_table["RFM_Score"] = rfm_table[["R", "F", "M"]].sum(axis = 1)

                # st.dataframe(rfm_table.head())
                segt_map = {
                r'[1-2][1-2]': 'Hibernating',
                r'[1-2][3-4]': 'At-Risk',
                r'[1-2]5': 'Cannot lose them',
                r'3[1-2]': 'About To Sleep',
                r'33': 'Need Attention',
                r'[3-4][4-5]': 'Loyal Customers',
                r'41': 'Promising',
                r'51': 'New Customers',
                r'[4-5][2-3]': 'Potential Loyalists',
                r'5[4-5]': 'Champions'
                }
                rfm_table['Segment'] = rfm_table['R'].astype(str) + rfm_table['F'].astype(str)
                rfm_table['Segment'] = rfm_table['Segment'].replace(segt_map, regex=True)
                st.dataframe(rfm_table.head())

                try:
                    filename = "Customer_Behaviour.xlsx"
                    # Create a BytesIO stream to save the Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        rfm_table.to_excel(writer, index=True, sheet_name='Sheet1')
                    output.seek(0)  # Seek to the beginning of the stream

                    # Provide the download button
                    st.download_button(
                        label="Download Excel",
                        data=output,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )
                except Exception as e:
                    logging.error("Error generating Excel file: %s", e)
                    st.error("An error occurred while generating the Excel file.")

                #description
                st.write("Recency: How recently a customer has made a purchase")
                st.write("Frequency: How often a customer makes a purchase")
                st.write("Monetary Value: How much money a customer spends on purchases")

                rfm_coordinates = {"Champions": [3, 5, 0.8, 1],
                            "Loyal Customers": [3, 5, 0.4, 0.8],
                            "Cannot lose them": [4, 5, 0, 0.4],
                            "At-Risk": [2, 4, 0, 0.4],
                            "Hibernating": [0, 2, 0, 0.4],
                            "About To Sleep": [0, 2, 0.4, 0.6],
                            "Promising": [0, 1, 0.6, 0.8],
                            "New Customers": [0, 1, 0.8, 1],
                            "Potential Loyalists": [1, 3, 0.6, 1],
                            "Need Attention": [2, 3, 0.4, 0.6]}

                #code for plot
                fig, ax = plt.subplots(figsize = (19, 15))

                ax.set_xlim([0, 5])
                ax.set_ylim([0, 5])

                plt.rcParams["axes.facecolor"] = "white"
                palette = ["#282828", "#04621B", "#971194", "#F1480F",  "#4C00FF", 
                        "#FF007B", "#9736FF", "#8992F3", "#B29800", "#80004C"]

                for key, color in zip(rfm_coordinates.keys(), palette[:10]):
                    
                    coordinates = rfm_coordinates[key]
                    ymin, ymax, xmin, xmax = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
                    
                    ax.axhspan(ymin = ymin, ymax = ymax, xmin = xmin, xmax = xmax, facecolor = color)
                    
                    users = rfm_table[rfm_table.Segment == key].shape[0]
                    users_percentage = (rfm_table[rfm_table.Segment == key].shape[0] / rfm_table.shape[0]) * 100
                    avg_monetary = rfm_table[rfm_table.Segment == key]["Monetary"].mean()
                    
                    user_txt = "\n\nTotal Users: " + str(users) + "(" +  str(round(users_percentage, 2)) + "%)"
                    monetary_txt = "\n\n\n\nAverage Monetary: " + str(round(avg_monetary, 2))
                    
                    x = 5 * (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    
                    plt.text(x = x, y = y, s = key, ha = "center", va = "center", fontsize = 18, color = "white", fontweight = "bold")
                    plt.text(x = x, y = y, s = user_txt, ha = "center", va = "center", fontsize = 14, color = "white")    
                    plt.text(x = x, y = y, s = monetary_txt, ha = "center", va = "center", fontsize = 14, color = "white")    
                    
                    ax.set_xlabel("Recency Score")
                    ax.set_ylabel("Frequency Score")
                    
                sns.despine(left = True, bottom = True)
                st.pyplot(fig)
                
                st.write("Champions: Bought recently, buy often and spend the most")
                st.write("Loyal customers: Buy on a regular basis. Responsive to promotions.")
                st.write("Potential loyalist: Recent customers with average frequency.")
                st.write("Recent customers: Bought most recently, but not often")
                st.write("Promising: Recent shoppers, but haven’t spent much")
                st.write("Needs attention: Above average recency, frequency and monetary values. May not have bought very recently though.")
                st.write("About to sleep: Below average recency and frequency. Will lose them if not reactivated")
                st.write("At risk: Some time since they’ve purchased. Need to bring them back!")
                st.write("Can’t lose them: Used to purchase frequently but haven’t returned for a long time.")
                st.write("Hibernating: Last purchase was long back and low number of orders. May be lost.")

            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                st.error(f"Error during analysis: {e}")

with tab5:
    st.title("Product Analysis")
    df  = None
    if 'data4' not in st.session_state:
        if st.button("Upload Dataset Product Analysis"):
            st.switch_page("Data_Upload.py")
    else:
        df = st.session_state.data4 

    if df is not None:
        data = df.copy()
        st.write("First few rows of the data:")
        st.dataframe(data.head())

        try:
            product = st.selectbox("Select your Product ID column", df.columns, index=None)
            Quantity = st.selectbox("Select Quantity column", df.columns, index =None)
            InvoiceNo = st.selectbox("Select column conatins Invoice Number", df.columns, index =None)
            
            
            if all([product, Quantity, InvoiceNo]):
                basket = (df.groupby([InvoiceNo, product])[Quantity]
                        .sum().unstack().fillna(0))

                # Convert quantities to binary 0/1 (presence/absence) for Apriori
                basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

                # Apply Apriori algorithm to find frequent itemsets
                frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)

                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                st.write("Association Rules:")
                st.dataframe(rules)

                # Filter rules with lift > 1
                filtered_rules = rules[rules['lift'] > 1]
                st.write("Filtered Rules (lift > 1):")
                st.dataframe(filtered_rules)
                # Final note
                st.write("Market basket analysis complete!")

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            st.error("This analysis is not available for this data")