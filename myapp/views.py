# views.py
import pandas as pd
import os
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import DateRangeForm, ClusterSelectionForm
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
import json
import plotly.graph_objects as go
from collections import Counter

model = tf.keras.models.load_model('myapp/model_coba10.h5')

activity_descriptions = {
    'HOMEPAGE': 'Mengakses halaman utama atau halaman home',
    'ITEM_DETAIL': 'Melihat detail barang',
    'SEARCH': 'Melakukan pencarian',
    'BOOKING': 'Melakukan pemesanan atau pembelian',
    'PROMO_PAGE': 'Mengakses halaman promo',
    'ADD_PROMO': 'Menambahkan promo',
    'ADD_TO_CART': 'Menambah barang ke keranjang',
}

def preprocess_data(data):
    data['event_time'] = pd.to_datetime(data['event_time'])
    data_sampled = data.sort_values(by=['session_id', 'event_time'])
    data_backup = data_sampled[~data_sampled['event_name'].isin(['CLICK', 'SCROLL'])]
    data_backup_cleaned = data_backup[data_backup['event_name'] != data_backup.groupby('session_id')['event_name'].shift()]

    le = LabelEncoder()
    data_backup_cleaned['event_name'] = le.fit_transform(data_backup_cleaned['event_name'])
    grouped = data_backup_cleaned.groupby('session_id')['event_name'].apply(list)

    sequences = [session for session in grouped]
    max_seq_length = 29
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post', value=-1)
    
    return X, le, data_backup_cleaned

def get_date_range(data):
    data['event_time'] = pd.to_datetime(data['event_time'])
    start_date = data['event_time'].min().strftime('%d %B %Y')
    end_date = data['event_time'].max().strftime('%d %B %Y')
    return start_date, end_date

def get_visitor_data(data):
    data['event_time'] = pd.to_datetime(data['event_time'])
    data['week'] = data['event_time'].dt.to_period('W').astype(str)
    visitor_data = data.groupby('week')['session_id'].nunique().reset_index()
    visitor_data.columns = ['week', 'visitors']

    visitor_data['start_date'] = visitor_data['week'].apply(lambda x: pd.Period(x, freq='W').start_time.strftime('%d %B %Y'))
    visitor_data['end_date'] = visitor_data['week'].apply(lambda x: pd.Period(x, freq='W').end_time.strftime('%d %B %Y'))
    
    visitor_data['week_number'] = range(1, len(visitor_data) + 1)
    visitor_data['week_label'] = visitor_data['week_number'].apply(lambda x: f"Minggu ke {x}")
    
    total_visitors = int(visitor_data['visitors'].sum())
    
    return visitor_data, total_visitors
    
def get_activity_data(data):
    data_filtered = data[~data['event_name'].isin(['CLICK', 'SCROLL'])]
    activity_data = data_filtered['event_name'].value_counts().reset_index()
    activity_data.columns = ['activity', 'count']
    activity_data['description'] = activity_data['activity'].map(activity_descriptions)
    return activity_data

def create_sankey_diagram_per_cluster(cluster_representations, results):
    cluster_diagrams = {}
    for cluster, sequence in cluster_representations.items():
        steps = sequence.split()
        limited_steps = steps[:8]

        labels = []
        source = []
        target = []
        value = []

        label_dict = {}
        label_id = 0

        for i in range(len(limited_steps) - 1):
            step_current = f"{limited_steps[i]}_{i}"
            step_next = f"{limited_steps[i+1]}_{i+1}"

            if step_current not in label_dict:
                label_dict[step_current] = label_id
                labels.append(limited_steps[i])
                label_id += 1
            if step_next not in label_dict:
                label_dict[step_next] = label_id
                labels.append(limited_steps[i+1])
                label_id += 1

            source.append(label_dict[step_current])
            target.append(label_dict[step_next])
            value.append(1)

        node_positions = [i / (len(labels) - 1) for i in range(len(labels))]

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                x=node_positions,
                y=[0.5] * len(labels)
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])

        fig.update_layout(
            title_text=f"User Journey Map pada Cluster {cluster}",
            font_size=10,
            height=300
        )

        cluster_data = results[results['Cluster'] == cluster]
        activity_counts = cluster_data['Predicted Next Event'].value_counts()
        activity_percentages = (activity_counts / activity_counts.sum()) * 100
        activity_percentage_dict = activity_percentages.to_dict()
        activity_counts_dict = activity_counts.to_dict()

        cluster_diagrams[cluster] = {
            'diagram': fig.to_html(full_html=False),
            'percentage': activity_percentage_dict,
            'counts': activity_counts_dict
        }

    return cluster_diagrams

def get_cluster_frequencies(results):
    cluster_counts = results['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    return cluster_counts.sort_values(by='cluster')

def get_prediction_frequencies(results):
    prediction_counts = results['Predicted Next Event'].value_counts().reset_index()
    prediction_counts.columns = ['event', 'count']
    total_predictions = prediction_counts['count'].sum()
    prediction_counts['percentage'] = prediction_counts['count'] / total_predictions * 100
    return prediction_counts, total_predictions

def create_features(sequences):
    features = []
    for seq in sequences:
        seq_length = len(seq)
        unique_events = len(set(seq))
        features.append([seq_length, unique_events] + seq)
    return np.array(features)

def cluster_sequences(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans

def get_cluster_representations(results, label_encoder):
    cluster_representations = {}
    grouped = results.groupby('Cluster')

    for cluster, group in grouped:
        sequences = group['Input Sequence'].tolist()
        predictions = group['Predicted Next Event'].tolist()
        
        combined_sequences = [f"{sequence} {prediction}" for sequence, prediction in zip(sequences, predictions)]
        sequence_counts = Counter(combined_sequences)
        most_common_sequence = sequence_counts.most_common(1)[0][0]
        cluster_representations[cluster] = most_common_sequence

    return cluster_representations

def upload_dataset(request):
    if request.method == 'POST':
        date_form = DateRangeForm(request.POST)
        cluster_form = ClusterSelectionForm(request.POST)
        
        if date_form.is_valid() and cluster_form.is_valid():
            start_date = date_form.cleaned_data['start_date']
            end_date = date_form.cleaned_data['end_date'] + pd.DateOffset(days=1) - pd.DateOffset(seconds=1) # Include the entire end date
            num_clusters = cluster_form.cleaned_data['num_clusters']
            
            dataset_path = os.path.join(settings.BASE_DIR, 'datasets', 'click_stream_new.csv')
            data = pd.read_csv(dataset_path)
            
            data['event_time'] = pd.to_datetime(data['event_time']).dt.tz_localize(None)  # Convert to tz-naive
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            data = data[(data['event_time'] >= start_date) & (data['event_time'] <= end_date)]
            
            X, label_encoder, data_backup_cleaned = preprocess_data(data)
            predictions = model.predict(X)
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_labels = label_encoder.inverse_transform(predicted_classes)
            
            sequences = [" ".join(label_encoder.inverse_transform(seq[seq != -1])) for seq in X]
            results = pd.DataFrame({
                'Input Sequence': sequences,
                'Predicted Next Event': predicted_labels
            })
            
            X_numeric = [label_encoder.transform(seq.split()) for seq in sequences]
            X_numeric = pad_sequences(X_numeric, maxlen=29, padding='post', value=-1)
            predictions_numeric = label_encoder.transform(predicted_labels)
            combined = np.hstack([X_numeric, predictions_numeric.reshape(-1, 1)])
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(combined)
            results['Cluster'] = kmeans.labels_
            results = results.sort_values(by='Cluster')
            results['Cluster'] = results['Cluster'].astype(str)
            
            request.session['results'] = results.to_dict('records')
            data['event_time'] = data['event_time'].astype(str)
            request.session['raw_data'] = data.to_dict('records')

            visitor_chart_data, total_visitors = get_visitor_data(data)
            activity_chart_data = get_activity_data(data)

            if isinstance(visitor_chart_data, list):
                visitor_chart_data = pd.DataFrame(visitor_chart_data)
            if isinstance(activity_chart_data, list):
                activity_chart_data = pd.DataFrame(activity_chart_data)

            request.session['visitor_chart_data'] = visitor_chart_data.to_dict('records')
            request.session['total_visitors'] = total_visitors
            request.session['activity_chart_data'] = activity_chart_data.to_dict('records')

            cluster_representations = get_cluster_representations(results, label_encoder)
            request.session['cluster_representations'] = cluster_representations

            sankey_cluster_plots = create_sankey_diagram_per_cluster(cluster_representations, results)
            request.session['sankey_cluster_plots'] = sankey_cluster_plots

            cluster_frequencies = get_cluster_frequencies(results)
            request.session['cluster_frequencies'] = cluster_frequencies.to_dict('records')

            prediction_frequencies, total_predictions = get_prediction_frequencies(results)
            request.session['prediction_frequencies'] = prediction_frequencies.to_dict('records')

            graph_div = create_cluster_representation_chart(cluster_representations)
            request.session['cluster_rep_chart'] = graph_div
            
            start_date_str, end_date_str = get_date_range(data)
            request.session['date_range'] = f"{start_date_str} - {end_date_str}"
            
            return redirect('dashboard')
    else:
        date_form = DateRangeForm()
        cluster_form = ClusterSelectionForm()
    
    return render(request, 'main_page/upload.html', {'date_form': date_form, 'cluster_form': cluster_form})

def create_cluster_representation_chart(cluster_representations):
    rows = []
    for cluster, sequence in cluster_representations.items():
        events = sequence.split()
        events = events[:8]
        for i, event in enumerate(events):
            rows.append({
                'cluster': cluster,
                'event_name': event,
                'activity_number': i + 1
            })

    df = pd.DataFrame(rows)
    y_mapping = {
        'ADD_PROMO': 1,
        'ADD_TO_CART': 2,
        'BOOKING': 3,
        'HOMEPAGE': 4,
        'CLICK': 5,
        'ITEM_DETAIL': 6,
        'SCROLL': 7,
        'SEARCH': 8,
        'PROMO_PAGE': 9
    }

    fig = go.Figure()
    y_offsets = {cluster: i * 0.1 for i, cluster in enumerate(df['cluster'].unique())}

    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data['activity_number'],
            y=[y_mapping.get(event, 0) + y_offsets[cluster] for event in cluster_data['event_name']],
            mode='markers+lines',
            marker=dict(
                size=10
            ),
            text=[event for event in cluster_data['event_name']],
            name=f'Cluster {cluster}',
        ))

    fig.update_layout(
        xaxis_title='Urutan Aktivitas',
        yaxis=dict(
            title='Aktivitas Pengguna',
            tickmode='array',
            tickvals=list(y_mapping.values()),
            ticktext=list(y_mapping.keys()),
            showticklabels=True,
            range=[0.5, 9.5]
        ),
        xaxis=dict(
            dtick=1
        ),
        showlegend=True
    )

    return fig.to_html(full_html=False)

def dashboard(request):
    results = request.session.get('results', None)
    raw_data = request.session.get('raw_data', None)
    visitor_chart_data = request.session.get('visitor_chart_data', None)
    activity_chart_data = request.session.get('activity_chart_data', None)
    cluster_rep_chart = request.session.get('cluster_rep_chart', None)
    date_range = request.session.get('date_range', None)

    if raw_data:
        raw_data_df = pd.DataFrame(raw_data)
        num_users = raw_data_df['session_id'].nunique()
        transaction_count = raw_data_df[raw_data_df['event_name'].isin(['BOOKING', 'Success'])].shape[0]
        revenue = raw_data_df[raw_data_df['event_name'] == 'BOOKING']['item_price'].sum()
                
        if activity_chart_data:
            for activity in activity_chart_data:
                activity['activity'] = activity['activity'].replace('_', ' ')
        
        context = {
            'results': results,
            'num_users': num_users,
            'transaction_count': transaction_count,
            'revenue': revenue,
            'visitor_chart_data': json.dumps(visitor_chart_data),
            'visitor_chart_data_dua': visitor_chart_data,
            'activity_chart_data': json.dumps(activity_chart_data),
            'activity_chart_data_dua': activity_chart_data,
            'cluster_rep_chart': cluster_rep_chart, 
            'date_range': date_range

        }
    else:
        context = {
            'results': results,
            'visitor_chart_data': json.dumps(visitor_chart_data),
            'activity_chart_data': json.dumps(activity_chart_data),
            'visitor_chart_data_dua': visitor_chart_data,
            'activity_chart_data_dua': activity_chart_data,
            'cluster_rep_chart': cluster_rep_chart,
            'date_range': date_range
        }
    
    return render(request, 'dashboard/dashboard.html', context)

def user_journey(request):
    results = request.session.get('results', None)
    recall = request.session.get('recall', None)
    visitor_chart_data = request.session.get('visitor_chart_data', None)
    activity_chart_data = request.session.get('activity_chart_data', None)
    cluster_frequencies = request.session.get('cluster_frequencies', None)
    cluster_frequencies_dua = pd.DataFrame(request.session.get('cluster_frequencies', None))
    prediction_frequencies = request.session.get('prediction_frequencies', None)
    prediction_frequencies_dua = pd.DataFrame(request.session.get('prediction_frequencies', None))
    cluster_representations = request.session.get('cluster_representations', None)
    cluster_rep_chart = request.session.get('cluster_rep_chart', None)
    sankey_cluster_plots = request.session.get('sankey_cluster_plots', None)
    date_range = request.session.get('date_range', None)
    raw_data = request.session.get('raw_data', None)
    
    if raw_data:
        raw_data_df = pd.DataFrame(raw_data)
        num_users = raw_data_df['session_id'].nunique()

    cluster_frequencies_dua['cluster'] = cluster_frequencies_dua['cluster'].astype(int)

    cluster_table_data = []
    for cluster, data in sankey_cluster_plots.items():
        activities = cluster_representations[str(cluster)].split()[:8]
        sequence = " -> ".join(activities)
        
        frequency = cluster_frequencies_dua.loc[cluster_frequencies_dua['cluster'] == int(cluster), 'count']
        if not frequency.empty:
            frequency = frequency.values[0]
        else:
            frequency = 0

        cluster_table_data.append({
            'cluster': cluster,
            'activity_sequence': sequence,
            'frequency': frequency
        })

    context = {
        'recall': recall,
        'visitor_chart_data': json.dumps(visitor_chart_data),
        'activity_chart_data': json.dumps(activity_chart_data),
        'cluster_frequencies': json.dumps(cluster_frequencies),
        'cluster_frequencies_dua': cluster_frequencies,
        'prediction_frequencies': json.dumps(prediction_frequencies),
        'prediction_frequencies_dua': prediction_frequencies,
        'cluster_representations': json.dumps(cluster_representations),
        'cluster_rep_chart': cluster_rep_chart,
        'sankey_cluster_plots': sankey_cluster_plots,
        'date_range': date_range,
        'cluster_table_data': cluster_table_data,
        'num_users': num_users,

    }
    
    return render(request, 'user_journey/user_journey.html', context)
