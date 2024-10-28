# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from null_value import imputing_na
import openai
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tab1 = st.tabs(["Recommended Stock"])

# Azure OpenAI API client
openai.api_key = "bd18995c51fa40e19e493df21c7ded81"
openai.api_base = "https://madhukar-kumar.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

# Function to get completion from Azure OpenAI
def get_completion(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",  # Replace with your actual model deployment name in Azure
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error("Error while calling OpenAI API: %s", e)
        return "Error generating insight."
    
# Function to generate insights based on DataFrame or plot description
def generate_insight(description: str) -> str:
    prompt = f"Provide a summary or insight in bullet points based on the following data or graph description: {description}"
    return get_completion(prompt)





# st.set_page_config(
#     page_title="Inventory Management",
#     layout='wide'
# )
st.title("Recommended Stock")
#------------------------------------------------------------------------------------------------------------------------
end_inv = None
if 'data2' not in st.session_state:
    if st.button("Upload Dataset for End Inventory"):
        st.switch_page("data_upload.py")
else:
    end_inv = st.session_state.data2
#----------------------------------------------------------------------------------------------------------------------
sales = None
if 'data4' not in st.session_state:
    if st.button("Upload Dataset for Sales"):
        st.switch_page("data_upload.py")
else:
    sales = st.session_state.data4
#---------------------------------------------------------------------------------------------------------------------
purchases = None
if 'data3' not in st.session_state:
    if st.button("Upload Dataset for Purcahse Final"):
        st.switch_page("data_upload.py")
else:
    purchases = st.session_state.data3
#------------------------------------------------------------------------------------------------------------------------
# Proceed with data analysis if all datasets are loaded

if all([sales is not None, purchases is not None, end_inv is not None]):
    try:
        # select columns
        ord_date = st.selectbox("Select column which contains Order Date of product", purchases.columns, index =None)
        sale_date = st.selectbox("Select column which contains Sales Date of product", sales.columns, index =None)
        s_brand = st.selectbox("Select column which contains Brand of product in sales data", sales.columns, index =None)
        # s_desc = st.selectbox("Select column which contains Description of product in sales data", sales.columns, index =None)
        s_Qnt = st.selectbox("Select column which contains Quantity of product in sales data", sales.columns, index =None)
        end_date = st.selectbox("Select column which contains end date in end inventory", end_inv.columns, index =None)
        end_qnt = st.selectbox("Select column which contains product Quantity in end inventory", end_inv.columns, index =None)
        p_brand = st.selectbox("Select column which contains brand of product in purchase data", purchases.columns, index =None)
        # p_desc = st.selectbox("Select column which contains description of product in purchase data", purchases.columns, index =None)
        st.write("Select any one only if present in the dataset: Recieving Date or Lead Time")
        rec_date = st.selectbox("Select column which contains Recieving Date of product(Optional)", purchases.columns, index =None)
        lead_time = st.selectbox("Select column which contains lead time of product(Optional)", purchases.columns, index =None)

        if ord_date is not None:
            purchases[ord_date] = pd.to_datetime(purchases[ord_date])
            if rec_date is not None:
                purchases[rec_date] = pd.to_datetime(purchases[rec_date])
            else:
                purchases[rec_date] = purchases[ord_date] + pd.Timedelta(days=7)

        if all([ord_date is not None, sale_date is not None, s_brand is not None,end_qnt is not None, s_Qnt is not None, end_date is not None, p_brand is not None]):

            sales[sale_date] = pd.to_datetime(sales[sale_date])
            # purchases[rec_date] = pd.to_datetime(purchases[rec_date])
            # purchases[ord_date] = pd.to_datetime(purchases[ord_date])
            sales[sale_date] = sales[sale_date].dt.date
            start_day = sales[sale_date].min()
            end_day = sales[sale_date].max()
            
            col1,col2,col3 =st.columns(3)
            with col1:
                st.write("Start Date of data :", start_day)
                st.write("End Date of data :", end_day)
            with col2:
                date_in = st.date_input("Select a start date", start_day)
            with col3:
                date_out = st.date_input("Select a end date", end_day)

            total_days = (date_out - date_in).days
            # Calculating Sales Velocity for each product
            sales = sales[(sales[sale_date]< date_out)&(sales[sale_date]> date_in)]
            
            sales_velocity = sales.groupby([s_brand]).agg(Total_Sales=(s_Qnt, 'sum')).reset_index()
            sales_velocity['Sales_Per_Day'] = sales_velocity['Total_Sales'] / total_days
            if lead_time is not None:
                purchases['Lead_Time'] = purchases[lead_time]
            else:
                purchases.loc[:, 'Lead_Time'] = (purchases[rec_date] - purchases[ord_date]).dt.days

            lead_times = purchases.groupby([p_brand]).agg(Avg_Lead_Time=('Lead_Time', 'mean')).reset_index()
                
            # Merging the data
            merged_data = pd.merge(sales_velocity, lead_times, on=[s_brand], how='left')

            # Calculating Optimal Stock Level
            merged_data['Optimal_Stock_Level'] = merged_data['Sales_Per_Day'] * merged_data['Avg_Lead_Time']

            # Calculating Safety Stock using maximum sales for each product
            max_sales = sales.groupby([s_brand]).agg(Max_Daily_Sales=(s_Qnt, 'max')).reset_index()
            merged_data = pd.merge(merged_data, max_sales, on=[s_brand], how='left')

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # std_sales = merged_data['Sales_Per_Day'].std()
            # merged_data['Safety_Stock'] = 1.65 * std_sales * merged_data['Avg_Lead_Time']
            
            
            
            
            merged_data['Safety_Stock'] = merged_data['Max_Daily_Sales'] - merged_data['Sales_Per_Day']
            merged_data['Recommended_Stock_Level'] = merged_data['Optimal_Stock_Level'] + merged_data['Safety_Stock']

            # Filtering products where Sales_Per_Day value is greater than Max_Daily_Sales value
            problematic_products = merged_data[merged_data['Sales_Per_Day'] > merged_data['Max_Daily_Sales']]

            # Getting the number of problematic products
            num_problematic_products = len(problematic_products)
                
            # Updating Max_Daily_Sales for problematic products
            merged_data.loc[merged_data['Sales_Per_Day'] > merged_data['Max_Daily_Sales'], 'Max_Daily_Sales'] = merged_data['Sales_Per_Day']

            # Updating Safety Stock and Recommended Stock Level after modifying Max_Daily_Sales
            merged_data['Safety_Stock'] = merged_data['Max_Daily_Sales'] - merged_data['Sales_Per_Day']
            merged_data['Recommended_Stock_Level'] = round(merged_data['Optimal_Stock_Level'] + merged_data['Safety_Stock'])
        

            # Sorting the data by Sales_Per_Day in descending order to get top products
            top_products = sales_velocity.sort_values(by='Sales_Per_Day', ascending=False).head(20)  # you can adjust the number as needed

            # Creating the bar plot
            fig1, ax1 = plt.subplots(figsize=(15, 7))
            sns.barplot(x=s_brand, y='Sales_Per_Day', data=top_products, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            plt.title('Daily Sales Velocity by Product')
            plt.xlabel('Product')
            plt.ylabel('Sales Per Day')
            st.header("Daily Sales")
            st.pyplot(fig1)

            # --------------------------------------------------------------------------------------

            end_inv[end_date] = pd.to_datetime(end_inv[end_date])
            latest_inventory_date = end_inv[end_date].max()
            current_inventory = end_inv[end_inv[end_date] == latest_inventory_date]

            # Summarizing the current stock levels by product.
            current_stock_levels = current_inventory.groupby([s_brand]).agg(Current_Stock=(end_qnt, 'sum')).reset_index()

            # Merging the current stock levels with the previously calculated data.
            final_data = pd.merge(merged_data, current_stock_levels, on=[s_brand], how='left')

            # Assume zero current stock for any products not present in the current inventory.
            final_data['Current_Stock'] = final_data['Current_Stock'].fillna(0)

            # Calculating how much of each product needs to be ordered if current stock is below recommended levels.
            final_data.insert(2, 'Order_Quantity', round(final_data['Recommended_Stock_Level'] - final_data['Current_Stock']))
            final_data['Recommendation'] = np.select(
            [final_data['Order_Quantity']*(-1) < 0, final_data['Current_Stock'] > final_data['Recommended_Stock_Level'] * 0.5],
            ['Increase Stock', 'Reduce Stock'],
            default='OK')

            final_data['Order_Quantity'] = final_data['Order_Quantity'].clip(lower=0)  # Setting negative order quantities to zero.

            # Creating a graph using Seaborn.
            if (final_data['Order_Quantity']>0).any():
                sns.set(style="whitegrid")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Order_Quantity', y=s_brand, data=final_data.sort_values('Order_Quantity', ascending=False).head(10))
                plt.title('Top 10 Products to Reorder')
                plt.xlabel('Quantity to Order')
                plt.ylabel('Product Description')
                st.header("Top products to RE-ORDER:")
                st.pyplot(fig2)
            else:
                st.header("There is no Product to reorder")
            #--------------------------------------------------------------------------------------------

            # Sorting the products by recommended stock level
            sorted_data = final_data.sort_values(by='Recommended_Stock_Level', ascending=False).head(10)  # To show the top 10 products

            # Creating a double-column bar chart
            bar_width = 0.35
            index = np.arange(len(sorted_data))

            fig3, ax3 = plt.subplots(figsize=(15, 10))
            bar1 = plt.bar(index, sorted_data['Current_Stock'], bar_width, label='Current Stock', color='b')
            bar2 = plt.bar([i + bar_width for i in index], sorted_data['Recommended_Stock_Level'], bar_width, label='Recommended Stock', color='r')

            # Setting the labels and title
            plt.xlabel('Product Description')
            plt.ylabel('Stock Quantity')
            plt.title('Top 10 Products(by Recommended Stock Levels): Current vs Recommended Stock Levels')
            plt.xticks([i + bar_width / 2 for i in index], sorted_data[s_brand], rotation=45, ha='right')
            plt.legend()

            # Displaying the chart
            plt.tight_layout()
            st.header("Over-Stock Products")
            st.pyplot(fig3)
            df_description = sorted_data[[s_brand,'Current_Stock','Recommended_Stock_Level']]
            insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
            st.write("Insight for Over-Stock:")
            st.write(insight)


        #---------------------------------------------------------------------------------------------
            # Filtering the data to show the top 10 products where the ordering quantity is highest
            top_products_to_order = final_data.nlargest(10, 'Order_Quantity')

            # Plotting the bars
            fig4, ax = plt.subplots(figsize=(12, 8))

            # Indexing for the bars
            ind = np.arange(len(top_products_to_order))

            # Width of the bars
            bar_width = 0.4

            # Plotting current stock and recommended stock side by side
            ax.barh(ind, top_products_to_order['Current_Stock'], bar_width, color='skyblue', label='Current Stock')
            ax.barh([i + bar_width for i in ind], top_products_to_order['Recommended_Stock_Level'], bar_width, color='orange', label='Recommended Stock')

            # Setting the y-axis labels to product descriptions
            ax.set(yticks=[i + bar_width for i in ind], yticklabels=top_products_to_order[s_brand], ylim=[2 * bar_width - 1, len(ind)])

            # Adding the legend
            ax.legend()

            # Adding labels and title
            ax.set_xlabel('Quantity')
            ax.set_title('Top 10 Products(by Order Quantity): Current vs Recommended Stock Levels')

            # Display the plot
            if (final_data['Order_Quantity']>0).any():
                plt.tight_layout()
                st.header("Under-Stock Products")
                st.pyplot(fig4)
                df_description = top_products_to_order[[s_brand,'Current_Stock','Recommended_Stock_Level']]
                insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                st.write("Insight for Under-Stock:")
                st.write(insight)

        #-------------------------------------------------------
            st.write("Product sorted by Order Quantity")
            st.dataframe(final_data[[s_brand,'Order_Quantity','Current_Stock','Recommended_Stock_Level','Recommendation']].sort_values('Order_Quantity', ascending=False))
            st.write("Product sorted by Current stock Quantity")
            st.dataframe(final_data[[s_brand,'Order_Quantity','Current_Stock','Recommended_Stock_Level','Recommendation']].sort_values('Current_Stock', ascending=False))

            df = final_data[final_data['Order_Quantity']>0][[s_brand,'Order_Quantity','Current_Stock','Recommended_Stock_Level','Recommendation']]

            try:
                filename = "Stock_suggestion.xlsx"
                # Create a BytesIO stream to save the Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
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
    except Exception as e:
        logging.error("An error occurred during data processing: %s", e)
        st.error("An error occurred while processing your data.")