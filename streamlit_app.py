import streamlit as st  # Import Streamlit library for building web app UI
import pandas as pd  # Import pandas for dataframe manipulation
import random  # Import random module for generating random values
from datetime import datetime, timedelta  # Import datetime utilities for timestamp manipulation
from sklearn.ensemble import IsolationForest  # Import Isolation Forest algorithm for anomaly detection
import os  # Import os module for file system operations
from io import StringIO  # Import StringIO for in-memory string buffer operations

# generate sample log data
def generate_logs():
    logs = []  # initialize empty list to store logs
    for i in range(1000):  # create 1000 log entries
        time = datetime.now() - timedelta(minutes=random.randint(0, 1440))  # generate a random timestamp within the last 24 hours
        if i < 900:  # create normal logs (first 900 entries)
            log = {  # create a dictionary with normal log data
                'timestamp': time,  # log timestamp
                'response_time': random.randint(50, 500),  # normal response time (50-500 ms)
                'status_code': random.choice([200, 201, 202, 204]),  # HTTP success codes
                'user': f'user_{random.randint(1, 100)}'  # random user ID 1 to 100
            }
        else:  # create anomalous logs (last 100 entries)
            log = {  # create a dictionary with anomalous log data
                'timestamp': time,  # log timestamp
                'response_time': random.randint(500, 10000),  # slow response time (500-10000 ms)
                'status_code': random.choice([500, 502, 503, 504]),  # HTTP error codes
                'user': f'user{random.randint(1, 100)}'  # random user ID 1 to 100
            }
        logs.append(log)  # append log dictionary to logs list

    return pd.DataFrame(logs)  # convert logs list to pandas DataFrame and return it

#################################################################################################

# detect anomalies
def detect_anomalies(df):
    df_features = df.copy()  # create a copy of the dataframe to avoid modifying original

    # Try to parse timestamp for time-based features
    # If timestamp is already a valid datetime, use it; otherwise, skip timestamp-based features
    has_valid_timestamp = False  # initialize flag to track if timestamps are valid
    if 'timestamp' in df_features.columns:  # check if timestamp column exists
        if not pd.api.types.is_datetime64_any_dtype(df_features["timestamp"]):  # check if timestamp is not already datetime type
            # Try to parse the timestamp
            df_features["timestamp"] = pd.to_datetime(  # attempt to convert timestamp column to datetime
                df_features["timestamp"],  # source column to convert
                errors="coerce",  # convert invalid timestamps to NaT instead of raising error
            )
        # Check if we have any valid timestamps after conversion
        if df_features["timestamp"].notna().sum() > 0:  # check if there are any non-null timestamps
            has_valid_timestamp = True  # set flag to true if valid timestamps exist
    
    # Extract time features if we have valid timestamps
    if has_valid_timestamp:  # if valid timestamps exist, extract time components
        df_features['hour'] = df_features['timestamp'].dt.hour  # extract hour from timestamp (0-23)
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek  # extract day of week from timestamp (0-6, Monday-Sunday)
        df_features['minute_of_day'] = df_features['timestamp'].dt.hour * 60 + df_features['timestamp'].dt.minute  # calculate minute of day (0-1439)
    else:  # if timestamps are invalid, use synthetic time features based on row index
        df_features['hour'] = (df_features.index % 24).astype(int)  # create synthetic hour values (0-23)
        df_features['day_of_week'] = ((df_features.index // 24) % 7).astype(int)  # create synthetic day of week values (0-6)
        df_features['minute_of_day'] = (df_features.index % 1440).astype(int)  # create synthetic minute of day values (0-1439)

    # calculate user-based features
    user_stats = df_features.groupby('user')['response_time'].agg(['mean', 'std', 'count'])  # calculate response time statistics (mean, std, count) for each user
    user_status = df_features.groupby('user')['status_code'].mean()  # calculate average status code for each user
    user_error_rate = (df_features.groupby('user')['status_code'].apply(lambda x: (x >= 400).sum()) / df_features.groupby('user')['status_code'].count())  # calculate error rate (% of status codes >= 400) for each user
    user_stats = user_stats.join(user_status)  # join average status code to user_stats
    user_stats = user_stats.join(user_error_rate.rename('user_error_rate'))  # join error rate to user_stats with column name 'user_error_rate'
    user_stats = user_stats.reset_index()  # convert user index to a column
    user_stats.columns = ['user', 'user_avg_response', 'user_std_response', 'user_request_count', 'user_avg_status', 'user_error_rate']  # rename columns for clarity

    # merge user stats back into main dataframe
    df_features = df_features.merge(user_stats, on='user', how='left')  # merge user statistics to each row based on user ID

    # FEATURE 1: Response time deviation from user average
    df_features['response_deviation'] = abs(df_features['response_time'] - df_features['user_avg_response'])  # calculate absolute difference between response time and user's average
    
    # FEATURE 2: Response time z-score (how many standard deviations from mean)
    df_features['user_std_response'] = df_features['user_std_response'].fillna(0)  # replace NaN values in standard deviation with 0 (for users with 1 request)
    df_features['response_zscore'] = df_features.apply(  # calculate z-score for each response time
        lambda row: abs(row['response_time'] - row['user_avg_response']) / row['user_std_response'] if row['user_std_response'] > 0 else 0,  # z-score = (value - mean) / std_dev, or 0 if std_dev is 0
        axis=1  # apply function to each row
    )

    # FEATURE 3: Global response time anomaly (compared to all logs)
    global_mean_response = df_features['response_time'].mean()  # calculate mean response time across all logs
    global_std_response = df_features['response_time'].std()  # calculate standard deviation of response times across all logs
    df_features['global_response_zscore'] = (df_features['response_time'] - global_mean_response) / global_std_response  # calculate global z-score for each response time

    # FEATURE 4: Time-based activity pattern (requests in same hour by user)
    user_hour_counts = df_features.groupby(['user', 'hour']).size().reset_index(name='requests_in_hour')  # count requests per user per hour
    avg_requests_per_hour = user_hour_counts['requests_in_hour'].mean()  # calculate average requests per hour across all user-hour combinations
    max_requests_per_hour = user_hour_counts['requests_in_hour'].max()  # calculate maximum requests per hour
    df_features = df_features.merge(  # merge request counts back to main dataframe
        user_hour_counts, on=['user', 'hour'], how='left'  # merge on user and hour columns
    )
    df_features['requests_in_hour'] = df_features['requests_in_hour'].fillna(0)  # replace NaN with 0 for hours with no requests
    df_features['hour_activity_anomaly'] = df_features['requests_in_hour'] > (avg_requests_per_hour + 2 * user_hour_counts['requests_in_hour'].std())  # flag if requests exceed average + 2 standard deviations

    # FEATURE 5: Status code anomaly (error codes)
    df_features['is_error'] = (df_features['status_code'] >= 400).astype(int)  # create binary flag: 1 if error code (>=400), 0 otherwise
    df_features['user_error_deviation'] = abs(df_features['is_error'] - df_features['user_error_rate'])  # calculate deviation from user's typical error rate

    # FEATURE 6: Abnormal hour for user (requests at unusual times)
    user_hour_distribution = df_features.groupby(['user', 'hour']).size() / df_features.groupby('user').size().values[0]  # calculate proportion of requests per hour per user
    df_features['is_off_hours'] = df_features['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)  # flag requests during off-hours (10 PM - 5 AM)

    # FEATURE 7: User request frequency anomaly
    df_features['request_freq_zscore'] = (df_features['user_request_count'] - df_features['user_request_count'].mean()) / (df_features['user_request_count'].std() + 1)  # calculate z-score of request frequency for each user

    # FEATURE 8: Combined response time metric (captures slow responses)
    df_features['slow_response'] = (df_features['response_time'] > global_mean_response + 2 * global_std_response).astype(int)  # flag if response time exceeds mean + 2 standard deviations

    feature_cols = [  # list of features to use for anomaly detection
        'response_time',        # raw response time in milliseconds
        'status_code',          # HTTP status code
        'hour',                 # hour of the day (0-23)
        'minute_of_day',        # minute of the day (0-1439)
        'user_request_count',   # number of requests by this user
        'user_avg_response',    # average response time for this user
        'response_deviation',   # deviation from user's average response time
        'user_avg_status',      # average status code for this user
        'response_zscore',      # z-score of response time for this user
        'global_response_zscore', # global z-score of response time
        'requests_in_hour',     # number of requests in this hour by this user
        'user_error_rate',      # error rate for this user
        'is_error',             # binary flag: is this response an error
        'user_error_deviation', # deviation from this user's typical error rate
        'is_off_hours',         # binary flag: is this request at off-hours
        'slow_response',        # binary flag: is response time abnormally slow
    ]

    features = df_features[feature_cols]  # extract only the features needed for model training

    # Calculate expected contamination based on error status codes and other anomalies
    estimated_anomalies = len(df_features[df_features['status_code'] >= 500])  # count severe error codes (500+)
    # Add anomalies from slow responses
    estimated_anomalies += len(df_features[df_features['slow_response'] == 1])  # add count of slow responses
    # Add anomalies from high z-scores
    estimated_anomalies += len(df_features[df_features['response_zscore'] > 3])  # add count of responses with very high z-scores (>3)
    
    estimated_contamination = max(0.05, min(0.5, estimated_anomalies / len(df_features)))  # calculate contamination rate, bounded between 5% and 50%

    # initialize Isolation Forest model
    model = IsolationForest(contamination=estimated_contamination, random_state=42)  # create model with calculated contamination rate and fixed random seed

    # train model on features
    model.fit(features)  # fit the model on the feature set

    # predict anomalies (-1 for anomaly, 1 for normal)
    predictions = model.predict(features)  # generate predictions for each row
    # map numeric predictions to string labels
    df_features['anomaly'] = pd.Series(predictions, index=df_features.index).map({1: 'normal', -1: 'anomaly'})  # convert numeric predictions to 'normal' or 'anomaly' labels

    return df_features  # return dataframe with anomaly labels

#################################################################################################

# generate styled report
def generate_html_report(df, anomalies):
    total = len(df)  # calculate total number of logs
    num_anomalies = len(anomalies)  # calculate number of anomalies detected
    anomaly_rate = (num_anomalies / total) * 100  # calculate anomaly percentage

    html = f"""  # start HTML string with f-string formatting
    <!DOCTYPE html>  <!-- HTML5 document declaration -->
    <html>  <!-- root HTML element -->
    <head>  <!-- header section with metadata -->
        <title>Log Anomaly Report</title>  <!-- page title -->
        <style>  <!-- CSS styling section -->
            body {{  <!-- body styling with double braces for f-string -->
                font-family: Arial, sans-serif;  <!-- set font family -->
                margin: 40px;  <!-- set page margins -->
                background-color: #f5f5f5;  <!-- set background color -->
            }}
            h1 {{  <!-- h1 heading styling -->
                color: #333;  <!-- set text color -->
                text-align: center;  <!-- center align text -->
            }}
            .metric {{  <!-- CSS class for metric boxes -->
                background: white;  <!-- white background -->
                padding: 20px;  <!-- internal spacing -->
                margin: 10px auto;  <!-- external spacing and center -->
                border-radius: 8px;  <!-- rounded corners -->
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);  <!-- subtle shadow effect -->
                width: 80%;  <!-- set width to 80% -->
            }}
            table {{  <!-- table styling -->
                border-collapse: collapse;  <!-- collapse table borders -->
                width: 100%;  <!-- full width -->
                background: white;  <!-- white background -->
                margin: 20px 0;  <!-- vertical margins -->
            }}
            th, td {{  <!-- styling for table headers and cells -->
                border: 1px solid #ddd;  <!-- light border -->
                padding: 12px;  <!-- internal spacing -->
                text-align: left;  <!-- left align text -->
            }}
            th {{  <!-- table header styling -->
                background-color: #fcaf50;  <!-- orange background -->
                color: white;  <!-- white text color -->
            }}
            tr:nth-child(even) {{  <!-- alternate row styling (every even row) -->
                background-color: #f9f9f9;  <!-- light gray background -->
            }}
            .anomaly {{  <!-- CSS class for anomaly highlighting -->
                color: red;  <!-- red text color -->
                font-weight: bold;  <!-- bold font weight -->
            }}
        </style>  <!-- end style section -->
    </head>  <!-- end header section -->
    <body>  <!-- start body section -->
        <h1>Log Anomaly Dashboard</h1>  <!-- main page heading -->
        <div class="metric">  <!-- metrics box container -->
            <h2>Summary Statistics</h2>  <!-- subheading -->
            <p><strong>Total logs:</strong> {total}</p>  <!-- display total logs count -->
            <p><strong>Normal logs:</strong> {total - num_anomalies}</p>  <!-- display normal logs count -->
            <p><strong>Anomalies detected:</strong> <span class="anomaly">{num_anomalies}</span></p>  <!-- display anomalies count in red -->
            <p><strong>Anomaly rate:</strong> {anomaly_rate:.1f}%</p>  <!-- display anomaly percentage (1 decimal place) -->
        </div>  <!-- end metrics box -->

        <div class="metric">  <!-- anomalies details box -->
            <h2>Anomalies Detected</h2>  <!-- subheading -->
            {anomalies.to_html(index=False, escape=False)}  <!-- convert anomalies dataframe to HTML table -->
        </div>  <!-- end anomalies box -->
        
        <div class="metric">  <!-- all logs box -->
            <h2>All Logs</h2>  <!-- subheading -->
            {df.to_html(index=False, escape=False)}  <!-- convert all logs dataframe to HTML table -->
        </div>  <!-- end logs box -->
        
    </body>  <!-- end body section -->
    </html>  <!-- end HTML document -->
    """  # end HTML string
    return html  # return the HTML string

# Streamlit app layout

st.title("AI-Enhanced Log Monitoring Dashboard")  # set page title

# initialize session state to persist data
if 'df' not in st.session_state:  # check if dataframe is already in session state
    st.session_state.df = None  # initialize dataframe as None
if 'result_df' not in st.session_state:  # check if result dataframe is in session state
    st.session_state.result_df = None  # initialize result dataframe as None
if 'anomalies' not in st.session_state:  # check if anomalies dataframe is in session state
    st.session_state.anomalies = None  # initialize anomalies dataframe as None
if 'uploaded_file_id' not in st.session_state:  # check if file ID is in session state
    st.session_state.uploaded_file_id = None  # initialize file ID as None
if 'detection_run' not in st.session_state:  # check if detection_run flag is in session state
    st.session_state.detection_run = False  # initialize detection_run flag as False

option = st.radio("Choose an option:", ["Generate Sample Logs", "Upload CSV Log File"])  # create radio button for user to choose mode

# if log generation option is selected
if option == "Generate Sample Logs":  # check if user selected generate mode
    st.session_state.uploaded_file_id = None  # reset upload ID when switching to generate mode
    
    if st.button("Generate Logs"):  # check if generate button was clicked
        st.session_state.df = generate_logs()  # call log generation function and store result
        st.session_state.result_df = None  # clear previous anomaly detection results
        st.session_state.anomalies = None  # clear previous anomalies
        st.success("Sample logs generated!")  # display success message
        st.write(st.session_state.df.head())  # display first 5 rows of generated logs
    elif st.session_state.df is not None:  # if no button click but data exists, show previous data
        st.write("Previously Generated Logs:")  # display label
        st.write(f"Total logs: {len(st.session_state.df)}")  # display total log count
        st.write(st.session_state.df.head())  # display first 5 rows of previously generated logs
        
# if upload option is selected
elif option == "Upload CSV Log File":  # check if user selected upload mode
    uploaded_file = st.file_uploader("Upload your CSV log file", type=["csv"])  # create file uploader widget for CSV files
    if uploaded_file is not None:  # check if file was uploaded
        current_file_id = id(uploaded_file)  # get unique ID of current uploaded file
        if st.session_state.uploaded_file_id != current_file_id:  # check if this is a new file upload
            st.session_state.df = pd.read_csv(uploaded_file)  # read uploaded CSV file into dataframe
            st.session_state.uploaded_file_id = current_file_id  # store file ID to track this upload
            st.session_state.result_df = None  # clear previous anomaly detection results
            st.session_state.anomalies = None  # clear previous anomalies
            st.success("File uploaded successfully!")  # display success message
        st.write(f"Total logs: {len(st.session_state.df)}")  # display total log count from uploaded file
        st.write(st.session_state.df.head())  # display first 5 rows of uploaded logs

if st.session_state.df is not None:  # check if dataframe has data
    col_detect, col_clear = st.columns(2)  # create two columns for buttons
    with col_detect:  # first column for detect button
        if st.button("Run Anomaly Detection"):  # check if anomaly detection button was clicked
            with st.spinner("Detecting anomalies..."):  # show spinner while processing
                st.session_state.result_df = detect_anomalies(st.session_state.df.copy())  # call anomaly detection function
                st.session_state.anomalies = st.session_state.result_df[st.session_state.result_df['anomaly'] == 'anomaly']  # filter to get only anomalies
                st.session_state.detection_run = True  # set flag to indicate detection has run
                st.success("Anomaly detection complete!")  # display success message
    with col_clear:  # second column for clear button
        if st.button("Clear Results"):  # check if clear button was clicked
            st.session_state.result_df = None  # clear result dataframe
            st.session_state.anomalies = None  # clear anomalies dataframe
            st.session_state.detection_run = False  # reset detection flag
            st.rerun()  # rerun the app to refresh display

if st.session_state.result_df is not None:  # check if results exist
    result_df = st.session_state.result_df  # store result dataframe in variable
    anomalies = st.session_state.anomalies  # store anomalies dataframe in variable
    
    total = len(result_df)  # calculate total number of logs
    num_anomalies = len(anomalies)  # calculate number of anomalies
    anomaly_rate = (num_anomalies / total) * 100  # calculate anomaly percentage

    # display metrics
    st.subheader("Summary Statistics")  # display subheading
    col1, col2, col3, col4 = st.columns(4)  # create 4 columns for metric display
    col1.metric("Total Logs", total)  # display total logs in first column
    col2.metric("Normal Logs", total - num_anomalies)  # display normal logs count in second column
    col3.metric("Anomalies Detected", num_anomalies)  # display anomalies count in third column
    col4.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")  # display anomaly percentage in fourth column

    # interactive log filters
    filter_option = st.selectbox("Filter logs:", ["All", "Anomalies", "Normal"])  # create dropdown for log filtering
    if filter_option == "Anomalies":  # if user selected anomalies filter
        display_df = anomalies  # set display dataframe to only anomalies
    elif filter_option == "Normal":  # if user selected normal filter
        display_df = result_df[result_df['anomaly'] == 'normal']  # filter result dataframe to only normal logs
    else:  # if user selected all filter
        display_df = result_df  # set display dataframe to all logs
    
    st.write("Filtered Logs:")  # display label
    st.dataframe(display_df)  # display filtered logs table

    # status code summary
    st.subheader("Status Code Summary")  # display subheading
    status_summary = result_df['status_code'].value_counts()  # count occurrences of each status code
    st.bar_chart(status_summary)  # display bar chart of status codes

    # download buttons for CSV and HTML report
    st.subheader("Download Reports")  # display subheading
    csv_buffer = StringIO()  # create in-memory buffer for CSV data
    result_df.to_csv(csv_buffer, index=False)  # write result dataframe to CSV buffer
    st.download_button(  # create CSV download button
        label = "Download CSV",  # button label
        data = csv_buffer.getvalue(),  # data to download (CSV content)
        file_name = "logs_with_anomalies.csv",  # filename for download
        mime = "text/csv"  # MIME type for CSV
    )

    html_report = generate_html_report(result_df, anomalies)  # generate HTML report
    st.download_button("Download HTML Report", html_report, "dashboard.html", "text/html")  # create HTML download button
