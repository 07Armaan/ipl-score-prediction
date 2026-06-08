from flask import Flask,render_template,request
import joblib
import pandas as pd

app = Flask(__name__)
final_pipeline = joblib.load("final_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    batting_team = request.form.get("batting_team")
    bowling_team = request.form.get("bowling_team")
    home_advantage = request.form.get("home_advantage")
    powerplay_runs = request.form.get("powerplay_runs")
    powerplay_wickets = request.form.get("powerplay_wickets")
    boundaries_pp = request.form.get("boundaries_pp")
    dot_balls_pp = request.form.get("dot_balls_pp")
    extras_pp = request.form.get("extras_pp")
    pitch_type = request.form.get("pitch_type")
    venue_avg_score = request.form.get("venue_avg_score")
    batting_team_form = request.form.get("batting_team_form")
    bowling_team_form = request.form.get("bowling_team_form")

    new_data_features_names = [
        "batting_team",	
        "bowling_team",
        "home_advantage",	"powerplay_runs",	"powerplay_wickets",	"boundaries_pp",	
        "dot_balls_pp",	
        "extras_pp",	
        "pitch_type",	"venue_avg_score",	"batting_team_form",	"bowling_team_form"
    ]

    new_data = [[
        batting_team,	
        bowling_team,
        home_advantage,	powerplay_runs,	powerplay_wickets,	boundaries_pp,	
        dot_balls_pp,	
        extras_pp,	
        pitch_type,	venue_avg_score,	batting_team_form,	bowling_team_form
    ]]
    new_data_df = pd.DataFrame(
        new_data,
        columns=new_data_features_names
    )
    new_data_pred = final_pipeline.predict(new_data_df)

    return render_template("index.html",prediction=f"The prediction of 1st innings runs: {round(new_data_pred[0])}")

if __name__ == "__main__":
    app.run(debug=True)