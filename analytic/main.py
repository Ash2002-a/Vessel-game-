import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os


class GameAnalyzer:
    def __init__(self, cursor_csv_path, vessel_csv_path=None):
        # Define column structures
        self.cursor_columns = [
            'TIMESTAMP', 'X_POSITION', 'Y_POSITION',
            'IS_CUTTING', 'SCORE', 'TIME_LEFT',
            'LEVEL', 'FIELD_OF_VIEW', 'DISTRACTION_ID',
            'DISTRACTION_TYPE', 'DISTRACTION_ACTION'
        ]

        self.vessel_columns = [
            'TIMESTAMP', 'VESSEL_ID', 'IS_CORRECT',
            'START_X', 'START_Y', 'END_X', 'END_Y',
            'CONTROL_POINT1_X', 'CONTROL_POINT1_Y',
            'CONTROL_POINT2_X', 'CONTROL_POINT2_Y',
            'PATH_POINTS', 'EVENT', 'IS_CUT', 'LEVEL', 'IS_INTERTWINED'
        ]

        # Initialize datasets
        self.load_cursor_data(cursor_csv_path)
        self.vessel_data_available = False

        if vessel_csv_path and os.path.exists(vessel_csv_path):
            self.load_vessel_data(vessel_csv_path)
            self.vessel_data_available = True

        self.create_results_directory()

    def create_results_directory(self):
        """Create timestamped directory for analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'analysis_results_{timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)

    def load_cursor_data(self, csv_path):
        """Load and process cursor tracking CSV data"""
        self.cursor_df = pd.read_csv(csv_path, names=self.cursor_columns)
        self.cursor_df['TIMESTAMP'] = pd.to_datetime(
            self.cursor_df['TIMESTAMP'])
        self.cursor_df['IS_CUTTING'] = self.cursor_df['IS_CUTTING'].fillna(
            'false')
        self.cursor_df['IS_CUTTING'] = (
            self.cursor_df['IS_CUTTING'].astype(str).str.lower() == 'true')

        # Calculate time differences
        self.cursor_df['TIME_DIFF'] = self.cursor_df['TIMESTAMP'].diff(
        ).dt.total_seconds()
        self.cursor_df['TIME_DIFF'] = self.cursor_df['TIME_DIFF'].fillna(0)

        # Mark different sessions
        self.cursor_df['SESSION'] = (
            self.cursor_df['TIME_LEFT'].diff() > 0).cumsum()

        print("\nCursor Data Info:")
        print(self.cursor_df.info())
        print("\nFirst few rows of cursor data:")
        print(self.cursor_df.head())

    def load_vessel_data(self, csv_path):
        """Load and process vessel tracking CSV data"""
        self.vessel_df = pd.read_csv(csv_path, names=self.vessel_columns)
        self.vessel_df['TIMESTAMP'] = pd.to_datetime(
            self.vessel_df['TIMESTAMP'])

        # Convert boolean columns
        self.vessel_df['IS_CORRECT'] = (
            self.vessel_df['IS_CORRECT'].astype(str).str.lower() == 'true')
        self.vessel_df['IS_CUT'] = (
            self.vessel_df['IS_CUT'].astype(str).str.lower() == 'true')
        self.vessel_df['IS_INTERTWINED'] = (
            self.vessel_df['IS_INTERTWINED'].astype(str).str.lower() == 'true')

        # Process vessel path points (convert from JSON string to list of points)
        if 'PATH_POINTS' in self.vessel_df.columns:
            import json
            # Use a lambda function to safely parse the JSON - return empty list if parsing fails
            self.vessel_df['PATH_POINTS_LIST'] = self.vessel_df['PATH_POINTS'].apply(
                lambda x: json.loads(x) if isinstance(
                    x, str) and x.strip() else []
            )
            print("Path points data successfully loaded!")
        else:
            print("No path points data found in the vessel tracking CSV.")
            self.vessel_df['PATH_POINTS_LIST'] = [[]]

        # Mark different sessions based on vessel creation patterns
        # A new session starts when a vessel is created within 1 second of a previous vessel being cut
        self.vessel_df = self.vessel_df.sort_values('TIMESTAMP')
        self.vessel_df['TIME_DIFF'] = self.vessel_df['TIMESTAMP'].diff(
        ).dt.total_seconds()

        # Join with cursor data to get session information
        merged_df = pd.merge_asof(
            self.vessel_df.sort_values('TIMESTAMP'),
            self.cursor_df[['TIMESTAMP', 'SESSION']].sort_values('TIMESTAMP'),
            on='TIMESTAMP',
            direction='nearest'
        )

        self.vessel_df['SESSION'] = merged_df['SESSION']

        print("\nVessel Data Info:")
        print(self.vessel_df.info())
        print("\nFirst few rows of vessel data:")
        print(self.vessel_df.head())

    def analyze_level_performance(self):
        """Analyze performance metrics for each level"""
        level_stats = {}

        for level in range(1, 8):
            level_data = self.cursor_df[self.cursor_df['LEVEL'] == level]
            sessions = level_data['SESSION'].unique()

            stats = {
                'completion_times': [],
                'scores': [],
                'accuracy': [],
                'movement_patterns': [],
                'distraction_reaction_times': [],  # For distraction analysis
                'vessel_stats': {
                    'correct_cuts': 0,
                    'wrong_cuts': 0,
                    'missed_cuts': 0,
                    'total_correct_vessels': 0,
                    'total_wrong_vessels': 0
                }
            }

            for session in sessions:
                session_data = level_data[level_data['SESSION'] == session]

                # Calculate metrics from cursor data
                completion_time = session_data['TIME_DIFF'].sum()
                final_score = session_data['SCORE'].max()
                cuts = (session_data['IS_CUTTING'].diff() == 1).sum()

                # Default values if we don't have vessel data
                correct_cuts = final_score // 10
                wrong_cuts = abs(min(0, final_score)) // 5

                # If vessel data is available, use it for more accurate stats
                if self.vessel_data_available:
                    vessel_session_data = self.vessel_df[
                        (self.vessel_df['LEVEL'] == level) &
                        (self.vessel_df['SESSION'] == session)
                    ]

                    # Count correct and incorrect cuts
                    cut_events = vessel_session_data[vessel_session_data['EVENT'] == 'cut']
                    correct_cuts = cut_events[cut_events['IS_CORRECT']].shape[0]
                    wrong_cuts = cut_events[~cut_events['IS_CORRECT']].shape[0]

                    # Count missed vessels (correct vessels that weren't cut)
                    created_vessels = vessel_session_data[vessel_session_data['EVENT'] == 'created']
                    correct_vessels = created_vessels[created_vessels['IS_CORRECT']]

                    # Get vessel IDs that were cut
                    cut_vessel_ids = cut_events['VESSEL_ID'].unique()

                    # Find correct vessels that weren't cut
                    missed_cuts = 0
                    for _, vessel in correct_vessels.iterrows():
                        if vessel['VESSEL_ID'] not in cut_vessel_ids:
                            missed_cuts += 1

                    # Update vessel stats
                    stats['vessel_stats']['correct_cuts'] += correct_cuts
                    stats['vessel_stats']['wrong_cuts'] += wrong_cuts
                    stats['vessel_stats']['missed_cuts'] += missed_cuts
                    stats['vessel_stats']['total_correct_vessels'] += correct_vessels.shape[0]
                    stats['vessel_stats']['total_wrong_vessels'] += created_vessels[~created_vessels['IS_CORRECT']].shape[0]

                stats['completion_times'].append(completion_time)
                stats['scores'].append(final_score)
                total_cuts = correct_cuts + wrong_cuts
                stats['accuracy'].append(
                    correct_cuts / total_cuts if total_cuts > 0 else 0)
                stats['movement_patterns'].append(
                    self.analyze_movement_pattern(session_data))

                # Add distraction reaction times if available
                reaction_times = self.calculate_distraction_reaction_times(
                    session_data)
                if reaction_times:
                    stats['distraction_reaction_times'].extend(reaction_times)

            level_stats[level] = stats

        return level_stats

    def analyze_vessel_metrics(self):
        """Analyze vessel-specific metrics"""
        if not self.vessel_data_available:
            return None

        vessel_metrics = {
            'level_stats': {},
            'overall': {
                'correct_vessel_cut_rate': 0,
                'wrong_vessel_cut_rate': 0,
                'intertwined_accuracy': 0,
                'non_intertwined_accuracy': 0
            }
        }

        # Analyze per level
        for level in range(1, 8):
            level_data = self.vessel_df[self.vessel_df['LEVEL'] == level]

            # Skip if no data for this level
            if level_data.empty:
                continue

            metrics = {
                'total_vessels': level_data[level_data['EVENT'] == 'created'].shape[0],
                'correct_vessels': level_data[(level_data['EVENT'] == 'created') & (level_data['IS_CORRECT'])].shape[0],
                'wrong_vessels': level_data[(level_data['EVENT'] == 'created') & (~level_data['IS_CORRECT'])].shape[0],
                'cut_vessels': level_data[level_data['EVENT'] == 'cut'].shape[0],
                'correct_cuts': level_data[(level_data['EVENT'] == 'cut') & (level_data['IS_CORRECT'])].shape[0],
                'wrong_cuts': level_data[(level_data['EVENT'] == 'cut') & (~level_data['IS_CORRECT'])].shape[0],
                'is_intertwined': level_data['IS_INTERTWINED'].iloc[0] if not level_data.empty else False
            }

            # Calculate rates
            if metrics['correct_vessels'] > 0:
                metrics['correct_vessel_cut_rate'] = metrics['correct_cuts'] / \
                    metrics['correct_vessels']
            else:
                metrics['correct_vessel_cut_rate'] = 0

            if metrics['wrong_vessels'] > 0:
                metrics['wrong_vessel_cut_rate'] = metrics['wrong_cuts'] / \
                    metrics['wrong_vessels']
            else:
                metrics['wrong_vessel_cut_rate'] = 0

            # Calculate accuracy
            if metrics['cut_vessels'] > 0:
                metrics['accuracy'] = metrics['correct_cuts'] / \
                    metrics['cut_vessels']
            else:
                metrics['accuracy'] = 0

            vessel_metrics['level_stats'][level] = metrics

        # Calculate overall metrics
        if vessel_metrics['level_stats']:
            # Correct vessel cut rate
            total_correct_vessels = sum(m['correct_vessels']
                                        for m in vessel_metrics['level_stats'].values())
            total_correct_cuts = sum(m['correct_cuts']
                                     for m in vessel_metrics['level_stats'].values())

            if total_correct_vessels > 0:
                vessel_metrics['overall']['correct_vessel_cut_rate'] = total_correct_cuts / \
                    total_correct_vessels

            # Wrong vessel cut rate
            total_wrong_vessels = sum(m['wrong_vessels']
                                      for m in vessel_metrics['level_stats'].values())
            total_wrong_cuts = sum(m['wrong_cuts']
                                   for m in vessel_metrics['level_stats'].values())

            if total_wrong_vessels > 0:
                vessel_metrics['overall']['wrong_vessel_cut_rate'] = total_wrong_cuts / \
                    total_wrong_vessels

            # Intertwined vs non-intertwined accuracy
            intertwined_levels = [
                level for level, metrics in vessel_metrics['level_stats'].items()
                if metrics['is_intertwined']
            ]
            non_intertwined_levels = [
                level for level, metrics in vessel_metrics['level_stats'].items()
                if not metrics['is_intertwined']
            ]

            # Calculate accuracy for intertwined levels
            intertwined_cuts = sum(
                vessel_metrics['level_stats'][level]['cut_vessels']
                for level in intertwined_levels
            )
            intertwined_correct_cuts = sum(
                vessel_metrics['level_stats'][level]['correct_cuts']
                for level in intertwined_levels
            )

            if intertwined_cuts > 0:
                vessel_metrics['overall']['intertwined_accuracy'] = intertwined_correct_cuts / intertwined_cuts

            # Calculate accuracy for non-intertwined levels
            non_intertwined_cuts = sum(
                vessel_metrics['level_stats'][level]['cut_vessels']
                for level in non_intertwined_levels
            )
            non_intertwined_correct_cuts = sum(
                vessel_metrics['level_stats'][level]['correct_cuts']
                for level in non_intertwined_levels
            )

            if non_intertwined_cuts > 0:
                vessel_metrics['overall']['non_intertwined_accuracy'] = non_intertwined_correct_cuts / \
                    non_intertwined_cuts

        return vessel_metrics

    def calculate_distraction_reaction_times(self, data):
        """Calculate reaction times for distractions"""
        reaction_times = []

        # Get all unique distraction IDs that are not background distractions
        distraction_ids = data[
            (data['DISTRACTION_ID'].notna()) &
            (data['DISTRACTION_ID'] != 'background')
        ]['DISTRACTION_ID'].unique()

        for distraction_id in distraction_ids:
            # Get appear and click events for this distraction
            appear_event = data[
                (data['DISTRACTION_ID'] == distraction_id) &
                (data['DISTRACTION_ACTION'] == 'appear')
            ]

            click_event = data[
                (data['DISTRACTION_ID'] == distraction_id) &
                (data['DISTRACTION_ACTION'] == 'click')
            ]

            if not appear_event.empty and not click_event.empty:
                appear_time = appear_event['TIMESTAMP'].iloc[0]
                click_time = click_event['TIMESTAMP'].iloc[0]

                # Calculate reaction time in seconds
                reaction_time = (click_time - appear_time).total_seconds()

                if reaction_time > 0:  # Ensure valid reaction time
                    reaction_times.append({
                        'distraction_id': distraction_id,
                        'distraction_type': appear_event['DISTRACTION_TYPE'].iloc[0],
                        'reaction_time': reaction_time,
                        'appear_time': appear_time,
                        'click_time': click_time
                    })

        return reaction_times

    def analyze_background_distractions(self):
        """Analyze the impact of background distractions on performance"""
        # Find all background distraction events
        bg_start_events = self.cursor_df[
            (self.cursor_df['DISTRACTION_ID'] == 'background') &
            (self.cursor_df['DISTRACTION_ACTION'] == 'start')
        ]

        bg_end_events = self.cursor_df[
            (self.cursor_df['DISTRACTION_ID'] == 'background') &
            (self.cursor_df['DISTRACTION_ACTION'] == 'end')
        ]

        bg_periods = []

        for _, start_event in bg_start_events.iterrows():
            matching_end = bg_end_events[
                (bg_end_events['DISTRACTION_TYPE'] == start_event['DISTRACTION_TYPE']) &
                (bg_end_events['TIMESTAMP'] > start_event['TIMESTAMP'])
            ].iloc[0] if not bg_end_events[
                (bg_end_events['DISTRACTION_TYPE'] == start_event['DISTRACTION_TYPE']) &
                (bg_end_events['TIMESTAMP'] > start_event['TIMESTAMP'])
            ].empty else None

            if matching_end is not None:
                bg_periods.append({
                    'type': start_event['DISTRACTION_TYPE'],
                    'start_time': start_event['TIMESTAMP'],
                    'end_time': matching_end['TIMESTAMP'],
                    'duration': (matching_end['TIMESTAMP'] - start_event['TIMESTAMP']).total_seconds(),
                    'level': start_event['LEVEL'],
                    'session': start_event['SESSION']
                })

        # Calculate performance during vs. outside distraction periods
        results = {
            'calls': {'with_distraction': [], 'without_distraction': []},
            'heartRateAlerts': {'with_distraction': [], 'without_distraction': []},
            'voiceOvers': {'with_distraction': [], 'without_distraction': []}
        }

        for period in bg_periods:
            # Get all events during this distraction period
            during_distraction = self.cursor_df[
                (self.cursor_df['TIMESTAMP'] >= period['start_time']) &
                (self.cursor_df['TIMESTAMP'] <= period['end_time']) &
                (self.cursor_df['LEVEL'] == period['level']) &
                (self.cursor_df['SESSION'] == period['session'])
            ]

            # Get events outside distraction periods but in same session/level
            outside_distraction = self.cursor_df[
                ~((self.cursor_df['TIMESTAMP'] >= period['start_time']) &
                  (self.cursor_df['TIMESTAMP'] <= period['end_time'])) &
                (self.cursor_df['LEVEL'] == period['level']) &
                (self.cursor_df['SESSION'] == period['session'])
            ]

            # Calculate metrics (e.g., cutting accuracy, movement speed)
            if not during_distraction.empty:
                cut_attempts_during = (
                    during_distraction['IS_CUTTING'].diff() == 1).sum()
                score_change_during = during_distraction['SCORE'].max(
                ) - during_distraction['SCORE'].min()

                # Calculate movement speed
                x_diff = during_distraction['X_POSITION'].diff()
                y_diff = during_distraction['Y_POSITION'].diff()
                distances = np.sqrt(x_diff**2 + y_diff**2)
                speeds = distances / \
                    during_distraction['TIME_DIFF'].replace({0: np.nan})

                results[period['type']]['with_distraction'].append({
                    'cut_attempts': cut_attempts_during,
                    'score_change': score_change_during,
                    'avg_speed': speeds.mean(),
                    'duration': period['duration']
                })

            if not outside_distraction.empty:
                cut_attempts_outside = (
                    outside_distraction['IS_CUTTING'].diff() == 1).sum()
                score_change_outside = outside_distraction['SCORE'].max(
                ) - outside_distraction['SCORE'].min()

                # Calculate movement speed
                x_diff = outside_distraction['X_POSITION'].diff()
                y_diff = outside_distraction['Y_POSITION'].diff()
                distances = np.sqrt(x_diff**2 + y_diff**2)
                speeds = distances / \
                    outside_distraction['TIME_DIFF'].replace({0: np.nan})

                results[period['type']]['without_distraction'].append({
                    'cut_attempts': cut_attempts_outside,
                    'score_change': score_change_outside,
                    'avg_speed': speeds.mean(),
                    'duration': outside_distraction['TIME_DIFF'].sum()
                })

        return results, bg_periods

    def analyze_movement_pattern(self, data):
        """Analyze movement patterns for a session"""
        x_diff = data['X_POSITION'].diff()
        y_diff = data['Y_POSITION'].diff()
        distances = np.sqrt(x_diff**2 + y_diff**2)
        speeds = distances / data['TIME_DIFF'].replace({0: np.nan})

        return {
            'total_distance': distances.sum(),
            'average_speed': speeds.mean(),
            'max_speed': speeds.max(),
            'movement_count': len(data)
        }

    def plot_level_comparisons(self):
        """Create comparative visualizations for levels"""
        plt.figure(figsize=(15, 10))

        # Completion Times
        plt.subplot(2, 2, 1)
        self.plot_metric_by_level('TIME_DIFF', 'Completion Time by Level')

        # Scores
        plt.subplot(2, 2, 2)
        self.plot_metric_by_level('SCORE', 'Score Distribution by Level')

        # Movement Speed
        plt.subplot(2, 2, 3)
        self.plot_speed_by_level()

        # Cutting Accuracy
        plt.subplot(2, 2, 4)
        self.plot_accuracy_by_level()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir,
                    'plots', 'level_comparisons.png'))
        plt.close()

    def plot_vessel_metrics(self):
        """Plot vessel-specific metrics if data is available"""
        if not self.vessel_data_available:
            print("No vessel data available for plotting.")
            return

        vessel_metrics = self.analyze_vessel_metrics()
        if not vessel_metrics:
            return

        # Plot correct vs wrong vessel cut rates by level
        plt.figure(figsize=(12, 8))

        levels = sorted(vessel_metrics['level_stats'].keys())
        correct_cut_rates = [vessel_metrics['level_stats']
                             [level]['correct_vessel_cut_rate'] for level in levels]
        wrong_cut_rates = [vessel_metrics['level_stats']
                           [level]['wrong_vessel_cut_rate'] for level in levels]

        plt.bar([l - 0.2 for l in levels], correct_cut_rates,
                width=0.4, label='Correct Vessel Cut Rate', color='green')
        plt.bar([l + 0.2 for l in levels], wrong_cut_rates,
                width=0.4, label='Wrong Vessel Cut Rate', color='red')

        plt.xlabel('Level')
        plt.ylabel('Cut Rate')
        plt.title('Correct vs Wrong Vessel Cut Rates by Level')
        plt.xticks(levels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.results_dir,
                    'plots', 'vessel_cut_rates.png'))
        plt.close()

        # Plot accuracy comparison: intertwined vs non-intertwined vessels
        plt.figure(figsize=(10, 6))

        labels = ['Intertwined Vessels', 'Non-intertwined Vessels']
        accuracies = [
            vessel_metrics['overall']['intertwined_accuracy'] * 100,
            vessel_metrics['overall']['non_intertwined_accuracy'] * 100
        ]

        plt.bar(labels, accuracies, color=['orange', 'blue'])
        plt.ylabel('Accuracy (%)')
        plt.title('Cutting Accuracy: Intertwined vs Non-intertwined Vessels')
        plt.ylim(0, 100)

        # Add value labels on bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')

        plt.savefig(os.path.join(self.results_dir,
                    'plots', 'intertwined_comparison.png'))
        plt.close()

        # Plot vessel data heatmap by level
        plt.figure(figsize=(14, 8))

        # Prepare data for heatmap
        heatmap_data = []
        for level in levels:
            stats = vessel_metrics['level_stats'][level]
            row = {
                'Level': level,
                'Total Vessels': stats['total_vessels'],
                'Correct Vessels': stats['correct_vessels'],
                'Wrong Vessels': stats['wrong_vessels'],
                'Cut Vessels': stats['cut_vessels'],
                'Correct Cuts': stats['correct_cuts'],
                'Wrong Cuts': stats['wrong_cuts'],
                'Accuracy (%)': stats['accuracy'] * 100
            }
            heatmap_data.append(row)

        df_heatmap = pd.DataFrame(heatmap_data).set_index('Level')

        # Create heatmap
        sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Vessel Metrics by Level')
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_dir,
                    'plots', 'vessel_heatmap.png'))
        plt.close()

    def plot_distraction_reaction_times(self):
        """Plot reaction times for distractions"""
        # Collect all distraction reaction times
        reaction_times = []

        for level in range(3, 8):  # Distractions start at level 3
            level_data = self.cursor_df[self.cursor_df['LEVEL'] == level]
            sessions = level_data['SESSION'].unique()

            for session in sessions:
                session_data = level_data[level_data['SESSION'] == session]
                session_reactions = self.calculate_distraction_reaction_times(
                    session_data)
                reaction_times.extend(session_reactions)

        if not reaction_times:
            print("No distraction reaction time data available.")
            return

        # Convert to DataFrame for easier analysis
        reaction_df = pd.DataFrame(reaction_times)

        # Plot reaction times by distraction type
        plt.figure(figsize=(12, 6))

        sns.boxplot(x='distraction_type', y='reaction_time', data=reaction_df)
        plt.title('Reaction Time by Distraction Type')
        plt.xlabel('Distraction Type')
        plt.ylabel('Reaction Time (seconds)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots',
                    'distraction_reaction_times.png'))
        plt.close()

        # Plot reaction times by level
        reaction_df['level'] = reaction_df.apply(
            lambda row: self.cursor_df[(self.cursor_df['TIMESTAMP'] == row['appear_time']) &
                                       (self.cursor_df['DISTRACTION_ID'] == row['distraction_id'])]['LEVEL'].iloc[0],
            axis=1
        )

        plt.figure(figsize=(12, 6))

        sns.boxplot(x='level', y='reaction_time', data=reaction_df)
        plt.title('Reaction Time by Game Level')
        plt.xlabel('Level')
        plt.ylabel('Reaction Time (seconds)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots',
                    'distraction_reaction_by_level.png'))
        plt.close()

        return reaction_df

    def plot_background_distraction_impact(self, bg_results):
        """Plot the impact of background distractions on performance"""
        if not bg_results:
            print("No background distraction data available.")
            return

        # Prepare data for plotting
        distraction_types = []
        with_dist_speeds = []
        without_dist_speeds = []
        with_dist_scores = []
        without_dist_scores = []

        for dist_type, data in bg_results.items():
            if data['with_distraction'] and data['without_distraction']:
                distraction_types.append(dist_type)

                # Average speeds
                with_speed = np.mean(
                    [d['avg_speed'] for d in data['with_distraction'] if not np.isnan(d['avg_speed'])])
                without_speed = np.mean(
                    [d['avg_speed'] for d in data['without_distraction'] if not np.isnan(d['avg_speed'])])
                with_dist_speeds.append(with_speed)
                without_dist_speeds.append(without_speed)

                # Score changes normalized by duration
                with_score = np.sum([d['score_change'] for d in data['with_distraction']]) / \
                    np.sum([d['duration'] for d in data['with_distraction']])
                without_score = np.sum([d['score_change'] for d in data['without_distraction']]) / \
                    np.sum([d['duration']
                           for d in data['without_distraction']])
                with_dist_scores.append(with_score)
                without_dist_scores.append(without_score)

        if not distraction_types:
            print("Insufficient background distraction comparison data.")
            return

        # Plot movement speed comparison
        plt.figure(figsize=(12, 6))

        x = np.arange(len(distraction_types))
        width = 0.35

        plt.bar(x - width/2, with_dist_speeds, width, label='With Distraction')
        plt.bar(x + width/2, without_dist_speeds,
                width, label='Without Distraction')

        plt.xlabel('Distraction Type')
        plt.ylabel('Average Movement Speed (pixels/s)')
        plt.title('Impact of Background Distractions on Movement Speed')
        plt.xticks(x, [t.replace('Rate', ' Rate').replace(
            'Overs', ' Overs') for t in distraction_types])
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots',
                    'distraction_speed_impact.png'))
        plt.close()

        # Plot score change comparison
        plt.figure(figsize=(12, 6))

        plt.bar(x - width/2, with_dist_scores, width, label='With Distraction')
        plt.bar(x + width/2, without_dist_scores,
                width, label='Without Distraction')

        plt.xlabel('Distraction Type')
        plt.ylabel('Score Change per Second')
        plt.title('Impact of Background Distractions on Scoring Rate')
        plt.xticks(x, [t.replace('Rate', ' Rate').replace(
            'Overs', ' Overs') for t in distraction_types])
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots',
                    'distraction_score_impact.png'))
        plt.close()

    def plot_metric_by_level(self, metric, title):
        """Plot boxplot of metric across levels"""
        data = []
        labels = []

        for level in range(1, 8):
            level_data = self.cursor_df[self.cursor_df['LEVEL']
                                        == level][metric]
            if not level_data.empty:
                data.append(level_data)
                labels.append(f'Level {level}')

        plt.boxplot(data, labels=labels)
        plt.title(title)
        plt.xticks(rotation=45)

    def plot_speed_by_level(self):
        """Plot speed distribution for each level"""
        speeds = []
        labels = []

        for level in range(1, 8):
            level_data = self.cursor_df[self.cursor_df['LEVEL'] == level]
            if not level_data.empty:
                x_diff = level_data['X_POSITION'].diff()
                y_diff = level_data['Y_POSITION'].diff()
                distances = np.sqrt(x_diff**2 + y_diff**2)
                level_speeds = distances / \
                    level_data['TIME_DIFF'].replace({0: np.nan})
                speeds.append(level_speeds.dropna())
                labels.append(f'Level {level}')

        plt.boxplot(speeds, labels=labels)
        plt.title('Movement Speed by Level')
        plt.xticks(rotation=45)

    def plot_accuracy_by_level(self):
        """Plot cutting accuracy for each level"""
        accuracies = []
        labels = []

        for level in range(1, 8):
            level_data = self.cursor_df[self.cursor_df['LEVEL'] == level]
            sessions = level_data['SESSION'].unique()

            level_accuracies = []

            # If vessel data is available, use it for more accurate statistics
            if self.vessel_data_available:
                vessel_level_data = self.vessel_df[self.vessel_df['LEVEL'] == level]

                for session in sessions:
                    session_vessels = vessel_level_data[vessel_level_data['SESSION'] == session]

                    if not session_vessels.empty:
                        cut_events = session_vessels[session_vessels['EVENT'] == 'cut']
                        if not cut_events.empty:
                            correct_cuts = cut_events[cut_events['IS_CORRECT']].shape[0]
                            total_cuts = cut_events.shape[0]
                            accuracy = correct_cuts / total_cuts if total_cuts > 0 else 0
                            level_accuracies.append(accuracy)

            # Fallback to cursor data if no vessel data or no session data found
            if not level_accuracies:
                for session in sessions:
                    session_data = level_data[level_data['SESSION'] == session]
                    score = session_data['SCORE'].max()
                    correct_cuts = max(0, score) // 10
                    wrong_cuts = abs(min(0, score)) // 5
                    total_cuts = correct_cuts + wrong_cuts
                    accuracy = correct_cuts / total_cuts if total_cuts > 0 else 0
                    level_accuracies.append(accuracy)

            if level_accuracies:
                accuracies.append(level_accuracies)
                labels.append(f'Level {level}')

        plt.boxplot(accuracies, labels=labels)
        plt.title('Cutting Accuracy by Level')
        plt.xticks(rotation=45)

    def plot_cursor_path(self, level=None, session=None):
        """Plot cursor movement path for specific level/session"""
        data = self.cursor_df
        if level is not None:
            data = data[data['LEVEL'] == level]
        if session is not None:
            data = data[data['SESSION'] == session]

        plt.figure(figsize=(12, 8))

        # Plot movements
        non_cutting = data[~data['IS_CUTTING']]
        plt.plot(non_cutting['X_POSITION'], non_cutting['Y_POSITION'],
                 'b.', alpha=0.3, label='Moving')

        cutting = data[data['IS_CUTTING']]
        plt.plot(cutting['X_POSITION'], cutting['Y_POSITION'],
                 'r.', alpha=0.5, label='Cutting')

        # Plot distraction locations if available
        distractors = data[
            (data['DISTRACTION_ID'].notna()) &
            (data['DISTRACTION_ID'] != 'background') &
            (data['DISTRACTION_ACTION'] == 'appear')
        ]

        if not distractors.empty:
            plt.plot(distractors['X_POSITION'], distractors['Y_POSITION'],
                     'go', markersize=8, label='Distractions')

        # If vessel data is available, plot vessel endpoints
        if self.vessel_data_available:
            vessel_data = self.vessel_df
            if level is not None:
                vessel_data = vessel_data[vessel_data['LEVEL'] == level]
            if session is not None:
                vessel_data = vessel_data[vessel_data['SESSION'] == session]

            # Plot vessel start points
            created_vessels = vessel_data[vessel_data['EVENT'] == 'created']

            if not created_vessels.empty:
                # Correct vessels
                correct_vessels = created_vessels[created_vessels['IS_CORRECT']]
                if not correct_vessels.empty:
                    plt.plot(correct_vessels['START_X'], correct_vessels['START_Y'],
                             'co', markersize=8, label='Correct Vessel Start')
                    plt.plot(correct_vessels['END_X'], correct_vessels['END_Y'],
                             'c^', markersize=8, label='Correct Vessel End')

                # Incorrect vessels
                incorrect_vessels = created_vessels[~created_vessels['IS_CORRECT']]
                if not incorrect_vessels.empty:
                    plt.plot(incorrect_vessels['START_X'], incorrect_vessels['START_Y'],
                             'mo', markersize=8, label='Incorrect Vessel Start')
                    plt.plot(incorrect_vessels['END_X'], incorrect_vessels['END_Y'],
                             'm^', markersize=8, label='Incorrect Vessel End')

        title = 'Cursor Movement Path'
        if level is not None:
            title += f' - Level {level}'
        if session is not None:
            title += f' - Session {session}'

        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.results_dir, 'plots',
                                 f'cursor_path_L{level}_S{session}.png'))
        plt.close()

    def plot_vessel_heatmap(self, level=None):
        """Plot heatmap of vessel positions"""
        if not self.vessel_data_available:
            print("No vessel data available for heatmap.")
            return

        vessel_data = self.vessel_df
        if level is not None:
            vessel_data = vessel_data[vessel_data['LEVEL'] == level]

        if vessel_data.empty:
            print(f"No vessel data available for level {level}.")
            return

        plt.figure(figsize=(14, 10))

        # Create a 2D histogram/heatmap of vessel locations
        # We'll use both start and end points to create a more comprehensive view
        x_coords = np.concatenate(
            [vessel_data['START_X'].values, vessel_data['END_X'].values])
        y_coords = np.concatenate(
            [vessel_data['START_Y'].values, vessel_data['END_Y'].values])

        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=50, range=[[0, 800], [0, 600]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.imshow(heatmap.T, extent=extent, origin='lower',
                   cmap='hot', interpolation='nearest')
        plt.colorbar(label='Vessel Density')

        title = 'Vessel Position Heatmap'
        if level is not None:
            title += f' - Level {level}'

        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        plt.savefig(os.path.join(self.results_dir, 'plots',
                                 f'vessel_heatmap{"_L"+str(level) if level else ""}.png'))
        plt.close()

    def plot_vessel_paths(self, level=None, session=None, show_cursor_data=True):
        """Plot all vessel paths for a given level/session along with cursor data"""
        if not self.vessel_data_available:
            print("No vessel data available for path visualization.")
            return

        vessel_data = self.vessel_df

        # Filter by level if specified
        if level is not None:
            vessel_data = vessel_data[vessel_data['LEVEL'] == level]

        # Filter by session if specified
        if session is not None:
            vessel_data = vessel_data[vessel_data['SESSION'] == session]

        if vessel_data.empty:
            print(
                f"No vessel data available for the specified parameters (level={level}, session={session}).")
            return

        # Group by vessel ID to get unique vessels
        unique_vessels = vessel_data.groupby('VESSEL_ID').first().reset_index()

        # Only keep vessels with the 'created' event
        created_vessels = unique_vessels[unique_vessels['EVENT'] == 'created']

        if created_vessels.empty:
            print(f"No vessel creation data available for the specified parameters.")
            return

        plt.figure(figsize=(14, 10))

        # Draw vessels
        for idx, vessel in created_vessels.iterrows():
            # Color based on whether it's correct (green) or incorrect (red)
            color = 'g' if vessel['IS_CORRECT'] else 'r'
            alpha = 0.8
            linestyle = '-'

            # Draw the Bezier curve path if path points are available
            if not vessel['PATH_POINTS_LIST'] or len(vessel['PATH_POINTS_LIST']) < 2:
                # If no path points, just draw a line from start to end
                plt.plot([vessel['START_X'], vessel['END_X']], [vessel['START_Y'], vessel['END_Y']],
                         color=color, alpha=alpha, linestyle=linestyle, linewidth=2)
            else:
                # Extract all x and y coordinates from the path points
                path_points = vessel['PATH_POINTS_LIST']
                x_points = [pt['x'] for pt in path_points]
                y_points = [pt['y'] for pt in path_points]

                # Plot the path
                plt.plot(x_points, y_points, color=color, alpha=alpha,
                         linestyle=linestyle, linewidth=2,
                         label=f"{'Correct' if vessel['IS_CORRECT'] else 'Incorrect'} Vessel")

                # Mark start and end points
                plt.scatter([vessel['START_X'], vessel['END_X']], [vessel['START_Y'], vessel['END_Y']],
                            color=color, s=100, marker='o', alpha=0.8)

        # Add cursor data if requested and available
        if show_cursor_data:
            cursor_data = self.cursor_df

            # Apply the same filters
            if level is not None:
                cursor_data = cursor_data[cursor_data['LEVEL'] == level]
            if session is not None:
                cursor_data = cursor_data[cursor_data['SESSION'] == session]

            if not cursor_data.empty:
                # Sample cursor data to avoid overcrowding (take every 10th point)
                sampled_cursor = cursor_data.iloc[::10]

                # Split into cutting and non-cutting movements
                cutting = sampled_cursor[sampled_cursor['IS_CUTTING']]
                non_cutting = sampled_cursor[~sampled_cursor['IS_CUTTING']]

                # Plot cursor movements
                plt.scatter(non_cutting['X_POSITION'], non_cutting['Y_POSITION'],
                            color='blue', alpha=0.3, s=10, label='Cursor (moving)')
                plt.scatter(cutting['X_POSITION'], cutting['Y_POSITION'],
                            color='magenta', alpha=0.5, s=10, label='Cursor (cutting)')

        # Set up the plot
        plt.grid(True, alpha=0.3)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        title = 'Vessel Paths'
        if level is not None:
            title += f' - Level {level}'
        if session is not None:
            title += f' - Session {session}'

        plt.title(title)

        # Create legend without duplicate entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        # Set equal aspect ratio
        plt.axis('equal')

        # Set plot boundaries to match the game canvas
        plt.xlim(0, 800)
        plt.ylim(0, 600)

        # Save the plot
        filename = f'vessel_paths'
        if level is not None:
            filename += f'_L{level}'
        if session is not None:
            filename += f'_S{session}'

        plt.savefig(os.path.join(self.results_dir, 'plots', f'{filename}.png'))
        plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        level_stats = self.analyze_level_performance()
        bg_results, bg_periods = self.analyze_background_distractions()
        vessel_metrics = self.analyze_vessel_metrics(
        ) if self.vessel_data_available else None

        # Generate visualizations
        self.plot_distraction_reaction_times()
        self.plot_background_distraction_impact(bg_results)
        self.plot_level_comparisons()

        # Generate vessel-specific visualizations if data is available
        if self.vessel_data_available:
            self.plot_vessel_metrics()

            # Generate heatmaps for all vessels and per level
            self.plot_vessel_heatmap()
            for level in range(1, 8):
                self.plot_vessel_heatmap(level)

        report_path = os.path.join(self.results_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== Blood Vessel Game Analysis Report ===\n\n")

            for level in range(1, 8):
                if level not in level_stats:
                    continue

                stats = level_stats[level]
                f.write(f"\nLevel {level} Analysis:\n")
                f.write("-" * 20 + "\n")

                # Completion Time
                times = stats['completion_times']
                if times:
                    f.write(f"Completion Time (seconds):\n")
                    f.write(f"  Average: {np.mean(times):.2f}\n")
                    f.write(f"  Min: {np.min(times):.2f}\n")
                    f.write(f"  Max: {np.max(times):.2f}\n")

                # Scores
                scores = stats['scores']
                if scores:
                    f.write(f"\nScores:\n")
                    f.write(f"  Average: {np.mean(scores):.2f}\n")
                    f.write(f"  Best: {np.max(scores):.2f}\n")
                    f.write(f"  Worst: {np.min(scores):.2f}\n")

                # Accuracy
                accuracy = stats['accuracy']
                if accuracy:
                    f.write(f"\nAccuracy:\n")
                    f.write(f"  Average: {np.mean(accuracy)*100:.2f}%\n")

                # Vessel Statistics
                if self.vessel_data_available:
                    f.write("\nVessel Statistics:\n")
                    f.write(
                        f"  Correct Cuts: {stats['vessel_stats']['correct_cuts']}\n")
                    f.write(
                        f"  Wrong Cuts: {stats['vessel_stats']['wrong_cuts']}\n")
                    f.write(
                        f"  Missed Cuts (correct vessels not cut): {stats['vessel_stats']['missed_cuts']}\n")
                    f.write(
                        f"  Total Correct Vessels: {stats['vessel_stats']['total_correct_vessels']}\n")
                    f.write(
                        f"  Total Wrong Vessels: {stats['vessel_stats']['total_wrong_vessels']}\n")

                    if stats['vessel_stats']['total_correct_vessels'] > 0:
                        correct_cut_rate = stats['vessel_stats']['correct_cuts'] / \
                            stats['vessel_stats']['total_correct_vessels']
                        f.write(
                            f"  Correct Vessel Cut Rate: {correct_cut_rate:.2f}\n")

                    if stats['vessel_stats']['total_wrong_vessels'] > 0:
                        wrong_cut_rate = stats['vessel_stats']['wrong_cuts'] / \
                            stats['vessel_stats']['total_wrong_vessels']
                        f.write(
                            f"  Wrong Vessel Cut Rate: {wrong_cut_rate:.2f}\n")

                # Movement Patterns
                movements = stats['movement_patterns']
                if movements:
                    avg_distance = np.mean(
                        [m['total_distance'] for m in movements])
                    avg_speed = np.mean([m['average_speed']
                                        for m in movements])
                    f.write(f"\nMovement Patterns:\n")
                    f.write(f"  Average Distance: {avg_distance:.2f} pixels\n")
                    f.write(
                        f"  Average Speed: {avg_speed:.2f} pixels/second\n")

                # Distraction Reaction Times (starting from level 3)
                if level >= 3 and 'distraction_reaction_times' in stats and stats['distraction_reaction_times']:
                    reaction_times = [rt['reaction_time']
                                      for rt in stats['distraction_reaction_times']]
                    f.write(f"\nDistraction Reaction Times:\n")
                    f.write(
                        f"  Average: {np.mean(reaction_times):.2f} seconds\n")
                    f.write(f"  Min: {np.min(reaction_times):.2f} seconds\n")
                    f.write(f"  Max: {np.max(reaction_times):.2f} seconds\n")

                    # Breakdown by distraction type
                    by_type = {}
                    for rt in stats['distraction_reaction_times']:
                        dist_type = rt['distraction_type']
                        if dist_type not in by_type:
                            by_type[dist_type] = []
                        by_type[dist_type].append(rt['reaction_time'])

                    f.write(f"\n  By Distraction Type:\n")
                    for dist_type, times in by_type.items():
                        f.write(
                            f"    {dist_type}: {np.mean(times):.2f} seconds avg\n")

                f.write("\n" + "="*50 + "\n")

            # Vessel Metrics Overview (if available)
            if vessel_metrics:
                f.write("\nVessel Metrics Overview:\n")
                f.write("-" * 30 + "\n")

                f.write("\nOverall Statistics:\n")
                f.write(
                    f"  Correct Vessel Cut Rate: {vessel_metrics['overall']['correct_vessel_cut_rate']:.2f}\n")
                f.write(
                    f"  Wrong Vessel Cut Rate: {vessel_metrics['overall']['wrong_vessel_cut_rate']:.2f}\n")
                f.write(
                    f"  Intertwined Vessel Accuracy: {vessel_metrics['overall']['intertwined_accuracy']*100:.2f}%\n")
                f.write(
                    f"  Non-Intertwined Vessel Accuracy: {vessel_metrics['overall']['non_intertwined_accuracy']*100:.2f}%\n")

                f.write("\nEffect of Intertwined Vessels:\n")
                accuracy_diff = vessel_metrics['overall']['non_intertwined_accuracy'] - \
                    vessel_metrics['overall']['intertwined_accuracy']
                f.write(
                    f"  Accuracy Difference: {accuracy_diff*100:.2f}% (negative means intertwined is harder)\n")
                f.write("\n" + "="*50 + "\n")

            # Background distraction impact analysis
            if bg_periods:
                f.write("\nBackground Distraction Impact Analysis:\n")
                f.write("-" * 40 + "\n")

                f.write(
                    f"Total background distraction events: {len(bg_periods)}\n\n")

                for dist_type in bg_results:
                    if bg_results[dist_type]['with_distraction'] and bg_results[dist_type]['without_distraction']:
                        f.write(f"{dist_type} Impact:\n")

                        # Movement speed comparison
                        with_speed = np.mean([d['avg_speed'] for d in bg_results[dist_type]['with_distraction']
                                             if not np.isnan(d['avg_speed'])])
                        without_speed = np.mean([d['avg_speed'] for d in bg_results[dist_type]['without_distraction']
                                                if not np.isnan(d['avg_speed'])])

                        speed_change = (
                            (with_speed - without_speed) / without_speed) * 100 if without_speed != 0 else 0

                        f.write(
                            f"  Movement Speed: {speed_change:.1f}% change during distraction\n")
                        f.write(
                            f"    With distraction: {with_speed:.2f} pixels/second\n")
                        f.write(
                            f"    Without distraction: {without_speed:.2f} pixels/second\n")

                        # Score rate comparison
                        with_score = np.sum([d['score_change'] for d in bg_results[dist_type]['with_distraction']]) / \
                            np.sum(
                                [d['duration'] for d in bg_results[dist_type]['with_distraction']])
                        without_score = np.sum([d['score_change'] for d in bg_results[dist_type]['without_distraction']]) / \
                            np.sum(
                                [d['duration'] for d in bg_results[dist_type]['without_distraction']])

                        score_change = (
                            (with_score - without_score) / without_score) * 100 if without_score != 0 else 0

                        f.write(
                            f"  Scoring Rate: {score_change:.1f}% change during distraction\n")
                        f.write(
                            f"    With distraction: {with_score:.2f} points/second\n")
                        f.write(
                            f"    Without distraction: {without_score:.2f} points/second\n\n")


def main():
    try:
        # Check if vessel data exists
        vessel_path = '../data/vessel_tracking.csv'
        vessel_exists = os.path.exists(vessel_path)

        if vessel_exists:
            print("Vessel tracking data found. Including in analysis.")
            analyzer = GameAnalyzer('../data/cursor_tracking.csv', vessel_path)
        else:
            print(
                "No vessel tracking data found. Running analysis with cursor data only.")
            analyzer = GameAnalyzer('../data/cursor_tracking.csv')

        # Generate report
        analyzer.generate_report()

        # Generate per-level visualizations
        for level in range(1, 8):
            analyzer.plot_cursor_path(level=level)

            # Generate vessel path visualizations if data is available
            if vessel_exists:
                print(
                    f"Generating vessel path visualizations for level {level}...")
                analyzer.plot_vessel_paths(level=level)

                # Get unique sessions for this level
                level_data = analyzer.cursor_df[analyzer.cursor_df['LEVEL'] == level]
                sessions = level_data['SESSION'].unique()

                # For each session, create a detailed visualization
                for session in sessions:
                    print(
                        f"Generating vessel path visualizations for level {level}, session {session}...")
                    analyzer.plot_vessel_paths(level=level, session=session)

        print(f"\nAnalysis complete! Results saved in: {analyzer.results_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
