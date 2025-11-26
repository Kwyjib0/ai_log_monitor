import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import os
from io import StringIO

# generate sample log data
def generate_logs():
    # initialize empty list to store logs
    logs = []
    # create 1000 log entries
    for i in range(1000):
        # generate a random timestamp within the last 24 hours
        time = datetime.now() - timedelta(minutes=random.randint(0, 1440))
        # create normal logs
        if i < 900:
            log = {
                'timestamp': time, # log timestamp
                'response_time': random.randint(50, 500), # normal response time (50-500 ms)
                'status_code': random.choice([200, 201, 202, 204]), # HTTP success codes
                'user': f'user_{random.randint(1, 100)}' # random user ID 1 to 100
            }
        # create anomalous logs
        else:
            log = {
                'timestamp': time, # log timestamp
                'response_time': random.randint(500, 10000), # slow response time (500-10000 ms)
                'status_code': random.choice([500, 502, 503, 504]), # HTTP error codes
                'user': f'user{random.randint(1, 100)}' # random user ID 1 to 100
            }
        # append log dictionary to logs list
        logs.append(log)

    # convert logs list to pandas DataFrame
    return pd.DataFrame(logs)

#################################################################################################

# detect anomalies
def detect_anomalies(df):
    # select columns for anomaly detection
    features = df[['response_time', 'status_code']]

    # initialize Isolation Forest model with contamination set to 10%
    model = IsolationForest(contamination=0.1, random_state=42)

    # train model on features
    model.fit(features)

    # predict anomalies (-1 for anomaly, 1 for normal)
    df['anomaly'] = model.predict(features)

    # map numeric predictions to string labels
    df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})

    return df

#################################################################################################

# generate styled report
def generate_html_report(df, anomalies):
    # total number of logs
    total = len(df)
    # number of anomalies detected
    num_anomalies = len(anomalies)
    # calculate anomaly percentage
    anomaly_rate = (num_anomalies / total) * 100

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Log Anomaly Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .metric {{
                background: white;
                padding: 20px;
                margin: 10px auto;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                width: 80%;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background: white;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #fcaf50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .anomaly {{
                color: red;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>Log Anomaly Dashboard</h1>
        <div class="metric">
            <h2>Summary Statistics</h2>
            <p><strong>Total logs:</strong> {total}</p>
            <p><strong>Normal logs:</strong> {total - num_anomalies}</p>
            <p><strong>Anomalies detected:</strong> <span class="anomaly">{num_anomalies}</span></p>
            <p><strong>Anomaly rate:</strong> {anomaly_rate:.1f}%</p>
        </div>

        <div class="metric">
            <h2>Anomalies Detected</h2>
            {anomalies.to_html(index=False, escape=False)}
        </div>
        
        <div class="metric">
            <h2>All Logs</h2>
            {df.to_html(index=False, escape=False)}
        </div>
        
    </body>
    </html>
    """
    return html

# Streamlit app layout

st.title("AI-Enhanced Log Monitoring Dashboard")

# initialize session state to persist data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
# radio button for user choice
option = st.radio("Choose an option:", ["Generate Sample Logs", "Upload CSV Log File"])
# initialize empty dataframe
df = None

# if log generation option is selected
if option == "Generate Sample Logs":
    if st.button("Generate Logs"): # button to generate logs
        st.session_state.df = generate_logs() # call log generation function
        st.success("Sample logs generated!") # success message
        st.write(st.session_state.df.head()) # display first 5 rows of generated logs
        '''
        df = generate_logs() # call log generation function
        st.success("Sample logs generated!") # success message
        st.write(df.head()) # display first 5 rows of generated logs
        '''
    elif st.session_state.df is not None:
        st.write("Previously Generated Logs:")
        st.write(st.session_state.df.head()) # display previously generated logs
        
# if upload option is selected
elif option == "Upload CSV Log File":
    # file uploader widget
    uploaded_file = st.file_uploader("Upload your CSV log file", type=["csv"])
    if uploaded_file is not None: # check if file was uploaded
        st.session_state.df = pd.read_csv(uploaded_file) # read uploaded CSV file into dataframe
        st.success("File uploaded successfully!") # success message
        st.write(st.session_state.df.head()) # display first 5 rows of uploaded logs
        '''
        df = pd.read_csv(uploaded_file) # read uploaded CSV file into dataframe
        st.success("File uploaded successfully!") # success message
        st.write(df.head()) # display first 5 rows of uploaded logs
        '''

# if dataframe is not empty
'''
if df is not None:
    if st.button("Run Anomaly Detection"): # button to run anomaly detection
        result_df = detect_anomalies(df) # call anomaly detection function
'''
if st.session_state.df is not None:
    if st.button("Run Anomaly Detection"): # button to run anomaly detection        
        # result_df = detect_anomalies(st.session_state.df.copy()) # call anomaly detection function
        st.session_state.result_df = detect_anomalies(st.session_state.df.copy()) # call anomaly detection function
        st.session_state.anomalies = st.session_state.result_df[st.session_state.result_df['anomaly'] == 'anomaly'] # filter anomalies

if st.session_state.result_df is not None:
    result_df = st.session_state.result_df
    anomalies = st.session_state.anomalies
    # calculate statistics
    total = len(result_df) # total number of logs
    # anomalies = result_df[result_df['anomaly'] == 'anomaly'] # filter anomalies
    num_anomalies = len(anomalies) # count anomalies
    anomaly_rate = (num_anomalies / total) * 100 # calculate anomaly rate

    # display metrics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4) # create 4 columns for metrics
    col1.metric("Total Logs", total) # display total logs
    col2.metric("Normal Logs", total - num_anomalies) # display normal log count
    col3.metric("Anomalies Detected", num_anomalies) # display anomaly count
    col4.metric("Anomaly Rate", f"{anomaly_rate:.1f}%") # display anomaly rate

    # interactive log filters
    filter_option = st.selectbox("Filter logs:", ["All", "Anomalies", "Normal"]) # dropdown for log filtering
    '''
    filter_option = st.selectbox("Filter logs:", ["All", "Anomalies", "Normal"]) # dropdown for log filtering
    '''
    if filter_option == "Anomalies":
        display_df = anomalies # show only anomalies
    elif filter_option == "Normal":
        display_df = result_df[result_df['anomaly'] == 'normal'] # show only normal logs
    else:
        display_df = result_df # show all logs
    
    st.write("Filtered Logs:")
    st.dataframe(display_df) # display filtered logs

    # status code summary
    st.subheader("Status Code Summary")
    status_summary = result_df['status_code'].value_counts() # count occurrences of each status code
    st.bar_chart(status_summary) # display bar chart of status codes

    # download buttons for CSV and HTML report
    st.subheader("Download Reports")
    csv_buffer = StringIO() # create in-memory buffer for CSV
    result_df.to_csv(csv_buffer, index=False) # write dataframe to CSV buffer
    #CSV download button
    st.download_button(
        label = "Download CSV",
        data = csv_buffer.getvalue(),
        file_name = "logs_with_anomalies.csv",
        mime = "text/csv"
    )

    html_report = generate_html_report(result_df, anomalies) # generate HTML report
    st.download_button("Download HTML Report", html_report, "dashboard.html", "text/html") # HTML download button

    # Expandable section to view logs
    with st.expander("View All Logs"): # collapsible section
        st.dataframe(result_df) # display all logs
