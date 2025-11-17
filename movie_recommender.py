import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLineEdit, QPushButton, QListWidget, 
                             QLabel, QMessageBox, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor


class RecommendationWorker(QThread):
    """
    Worker thread to perform movie recommendation calculations in the background.
    This prevents the GUI from freezing during computation.
    """
    finished = pyqtSignal(list, str)  # Signal to emit results or error message
    error = pyqtSignal(str)  # Signal to emit error messages
    
    def __init__(self, movie_title, df, cosine_sim):
        super().__init__()
        self.movie_title = movie_title
        self.df = df
        self.cosine_sim = cosine_sim
    
    def run(self):
        """Execute the recommendation algorithm in the background thread."""
        try:
            # Get index of the movie from its title
            movie_matches = self.df[self.df['title'].str.lower() == self.movie_title.lower()]
            
            if movie_matches.empty:
                # Try case-insensitive partial match
                movie_matches = self.df[self.df['title'].str.lower().str.contains(
                    self.movie_title.lower(), na=False)]
            
            if movie_matches.empty:
                self.error.emit(f"Movie '{self.movie_title}' not found in the database.")
                return
            
            # Get the first match
            movie_index = movie_matches.index[0]
            
            # Get similarity scores for this movie
            similar_movies = list(enumerate(self.cosine_sim[movie_index]))
            
            # Sort by similarity score in descending order
            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
            
            # Get top 20 recommendations (excluding the movie itself)
            recommendations = []
            count = 0
            for movie in sorted_similar_movies:
                if movie[0] != movie_index:  # Exclude the movie itself
                    movie_title = self.df.iloc[movie[0]]['title']
                    similarity_score = movie[1]
                    recommendations.append((movie_title, similarity_score))
                    count += 1
                    if count >= 20:
                        break
            
            self.finished.emit(recommendations, None)
            
        except Exception as e:
            self.error.emit(f"An error occurred: {str(e)}")


class MovieRecommenderGUI(QMainWindow):
    """
    Main GUI window for the Movie Recommender application.
    Provides a modern, responsive interface for movie recommendations.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize data structures
        self.df = None
        self.cosine_sim = None
        self.worker = None
        
        # Load and process data
        self.load_and_process_data()
        
        # Setup UI
        self.init_ui()
    
    def load_and_process_data(self):
        """
        Load the movie dataset and compute cosine similarity matrix.
        This is done once at startup for efficiency.
        """
        try:
            # Step 1: Read the CSV file
            self.df = pd.read_csv("movie_dataset.csv")
            
            # Step 2: Select features for recommendation
            features = ['keywords', 'cast', 'genres', 'director']
            
            # Step 3: Fill NaN values and create combined features column
            for feature in features:
                self.df[feature] = self.df[feature].fillna('')
            
            def combine_features(row):
                """Combine all selected features into a single string."""
                try:
                    return (str(row['keywords']) + " " + 
                           str(row['cast']) + " " + 
                           str(row['genres']) + " " + 
                           str(row['director']))
                except Exception as e:
                    print(f"Error combining features: {e}")
                    return ""
            
            self.df["combined_features"] = self.df.apply(combine_features, axis=1)
            
            # Step 4: Create count matrix from combined features
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(self.df['combined_features'])
            
            # Step 5: Compute cosine similarity matrix
            self.cosine_sim = cosine_similarity(count_matrix)
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", 
                               "movie_dataset.csv not found. Please ensure the file is in the same directory.")
            sys.exit(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Error loading data: {str(e)}")
            sys.exit(1)
    
    def init_ui(self):
        """Initialize and setup the user interface."""
        # Set window properties
        self.setWindowTitle("Movie Recommender System")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(600, 400)  # Minimum window size for responsiveness
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title label
        title_label = QLabel("🎬 Movie Recommender System")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Subtitle label
        subtitle_label = QLabel("Enter a movie title to get personalized recommendations")
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #7f8c8d; margin-bottom: 15px;")
        main_layout.addWidget(subtitle_label)
        
        # Input section (horizontal layout)
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        
        # Movie title input label
        input_label = QLabel("Movie Title:")
        input_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #34495e;")
        input_layout.addWidget(input_label)
        
        # Text input box
        self.movie_input = QLineEdit()
        self.movie_input.setPlaceholderText("e.g., The Dark Knight, Avatar, Inception...")
        self.movie_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                font-size: 13px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
            }
        """)
        self.movie_input.returnPressed.connect(self.get_recommendations)  # Enter key support
        input_layout.addWidget(self.movie_input, stretch=1)
        
        # Recommend button
        self.recommend_button = QPushButton("Recommend")
        self.recommend_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 25px;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.recommend_button.clicked.connect(self.get_recommendations)
        input_layout.addWidget(self.recommend_button)
        
        main_layout.addLayout(input_layout)
        
        # Results label
        self.results_label = QLabel("Recommendations will appear here...")
        self.results_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-top: 5px;")
        main_layout.addWidget(self.results_label)
        
        # Scrollable list widget for recommendations
        self.recommendations_list = QListWidget()
        self.recommendations_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #ecf0f1;
                border-radius: 5px;
                background-color: white;
                padding: 5px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #ecf0f1;
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: #ebf5fb;
            }
            QListWidget::item:selected {
                background-color: #d5e8f7;
                color: #2c3e50;
            }
        """)
        self.recommendations_list.setAlternatingRowColors(True)
        main_layout.addWidget(self.recommendations_list, stretch=1)  # Stretch to fill available space
        
        # Status label at the bottom
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 10px; color: #95a5a6; padding-top: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Apply modern color scheme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)
    
    def get_recommendations(self):
        """
        Triggered when the Recommend button is clicked or Enter is pressed.
        Starts the recommendation process in a background thread.
        """
        movie_title = self.movie_input.text().strip()
        
        # Validate input
        if not movie_title:
            QMessageBox.warning(self, "Input Required", 
                              "Please enter a movie title.")
            return
        
        # Disable button and clear previous results
        self.recommend_button.setEnabled(False)
        self.recommendations_list.clear()
        self.status_label.setText("Searching for recommendations...")
        self.results_label.setText("Processing...")
        
        # Create and start worker thread
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.worker = RecommendationWorker(movie_title, self.df, self.cosine_sim)
        self.worker.finished.connect(self.display_recommendations)
        self.worker.error.connect(self.handle_error)
        self.worker.start()
    
    def display_recommendations(self, recommendations, error):
        """
        Display the recommended movies in the list widget.
        Called when the worker thread finishes successfully.
        """
        self.recommend_button.setEnabled(True)
        
        if error:
            self.handle_error(error)
            return
        
        if not recommendations:
            self.results_label.setText("No recommendations found.")
            self.status_label.setText("No results")
            return
        
        # Clear the list
        self.recommendations_list.clear()
        
        # Add recommendations to the list
        for idx, (title, score) in enumerate(recommendations, 1):
            # Format similarity score as percentage
            score_percent = f"{score * 100:.1f}%"
            
            # Create list item with formatted text
            item_text = f"{idx}. {title} (Similarity: {score_percent})"
            item = QListWidgetItem(item_text)
            
            # Set item properties for better display
            item.setData(Qt.UserRole, (title, score))
            
            self.recommendations_list.addItem(item)
        
        # Update labels
        self.results_label.setText(f"Found {len(recommendations)} recommendations")
        self.status_label.setText(f"Displaying {len(recommendations)} recommendations")
    
    def handle_error(self, error_message):
        """
        Handle errors that occur during recommendation processing.
        Displays error message to the user.
        """
        self.recommend_button.setEnabled(True)
        self.status_label.setText("Error occurred")
        
        # Show error message box
        QMessageBox.warning(self, "Error", error_message)
        
        # Update UI labels
        self.results_label.setText("Error: " + error_message)
        self.recommendations_list.clear()
    
    def resizeEvent(self, event):
        """
        Handle window resize events to maintain responsive layout.
        This ensures the UI adapts properly to different window sizes.
        """
        super().resizeEvent(event)
        # The layout managers handle responsiveness automatically,
        # but we can add custom logic here if needed


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    
    # Set application style for modern look
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = MovieRecommenderGUI()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
