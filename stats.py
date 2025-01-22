import numpy as np
import scipy.stats as stats
import json
from scipy.stats import ttest_ind

#TODO: Make the code readable and add comments
#TODO: Format print statements to be pretty and understandable in the terminal

def load_json() -> dict:
    with open("results.json") as file:
        results = json.load(file)

    return results

def get_confidence_interval_and_average(data) -> dict:
    """Takes a list of data and returns the average, standard deviation, len of data and the confidence interval."""
    if len(data) == 0:
        raise ValueError("The data list must not be empty.")
    data = np.array(data)
    # average
    avg = np.mean(data)
    
    # Standard deviation
    std_dev = np.std(data, ddof=1)  
    
    # Beregn standard error
    standard_error = std_dev / np.sqrt(len(data))
    
    # confidence interval (95%)
    t_value = stats.t.ppf(0.975, df=len(data) - 1)  # 0.975 for tosidet 95%
    margin_of_error = t_value * standard_error
    confidence_interval = (avg - margin_of_error, avg + margin_of_error)
    
    return {
        "average": avg,
        "s": std_dev,
        "n": len(data),
        "confidence_interval": confidence_interval
    }

def compare(obs, num_games, obs2, num_games2) -> float:
    """returns the p-value of the two ratios"""
    if obs == 0 and obs2 == 0:
        return -999
    # calculate the ratio for each player
    ratio = obs / num_games
    ratio2 = obs2 / num_games2
    # calc the standard error for each player
    se = np.sqrt(ratio * (1 - ratio) / num_games)
    se2 = np.sqrt(ratio2 * (1 - ratio2) / num_games2)
    # calc the z-value
    z = (ratio - ratio2) / np.sqrt(se**2 + se2**2)
    #get p value
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p

def perform_t_test(list1, list2) -> tuple:
    """
    Perform a two-sample t-test to compare the means of two lists.
    """
    # Perform the t-test
    t_stat, p_value = ttest_ind(list1, list2, equal_var=False)  # Use Welch's t-test

    # Interpret the p-value
    significance_level = 0.05
    if p_value < significance_level:
        interpretation = "The difference between the two groups is statistically significant."
    else:
        interpretation = "The difference between the two groups is not statistically significant."

    # Return the results as a dictionary
    return t_stat, p_value, interpretation
    
from scipy.stats import t

def compare_means(avg1, std1, n1, avg2, std2, n2, confidence=0.95) -> tuple:
    """
    Compare two means using a confidence interval and a t-test.
    """
    # Calculate standard errors
    se1 = std1 / (n1 ** 0.5)
    se2 = std2 / (n2 ** 0.5)

    # Calculate the standard error of the difference
    se_diff = (se1**2 + se2**2) ** 0.5

    # Calculate the difference in means
    mean_diff = avg1 - avg2

    # Degrees of freedom for Welch's t-test
    df = ((se1**2 + se2**2)**2) / ((se1**4 / (n1 - 1)) + (se2**4 / (n2 - 1)))

    # Calculate the critical t-value
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha / 2, df)

    # Confidence interval for the difference
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff

    # Perform the t-test
    t_stat = mean_diff / se_diff
    p_value = 2 * t.sf(abs(t_stat), df)

    # Interpretation
    if p_value < alpha:
        interpretation = "The difference between the two groups is statistically significant."
    else:
        interpretation = "The difference between the two groups is not statistically significant."

    # Return results
    return mean_diff, (ci_lower, ci_upper), t_stat, p_value, interpretation

def compare_versions(dict1, dict2, alpha = 0.05) -> None:
    
    # calculate confidence level
    conf = 1 - alpha
    
    # calculate the ratio for white, black and draws
    for type in ["Stockfish", "Stockfish 1", "Stockfish 2", "Worstfish", "Random"]:
        print(type)
        white_p = compare(dict1[type]["White"], dict1["NumGames"], dict2[type]["White"], dict2["NumGames"])
        black_p = compare(dict1[type]["Black"], dict1["NumGames"], dict2[type]["Black"], dict2["NumGames"])
        draw_p = compare(dict1[type]["Draw"], dict1["NumGames"], dict2[type]["Draw"], dict2["NumGames"])
        if white_p < alpha:
            print("White: The difference between the two groups is statistically significant.")
        else: 
            print("White: The difference between the two groups is not statistically significant.")
        if black_p < alpha:
            print("Black: The difference between the two groups is statistically significant.")
        else:
            print("Black: The difference between the two groups is not statistically significant.")
        if draw_p < alpha:
            print("Draw: The difference between the two groups is statistically significant.")
        else:
            print("Draw: The difference between the two groups is not statistically significant.")
        
        # calculate t-test of move counts
        print("MoveCount")
        t_stat, p_val, interpretation = perform_t_test(dict1[type]["MoveCounts"], dict2[type]["MoveCounts"])
        print(t_stat, p_val, interpretation)
        
        # calculate t-test of Move speed
        print("MoveSpeed")
        mean_diff, ci, t_stat1, p_value, interpretation1 = compare_means(dict1[type]["AIMoveTimes"]["average"],
                                                                         dict1[type]["AIMoveTimes"]["s"],
                                                                         dict1[type]["AIMoveTimes"]["n"],
                                                                         dict2[type]["AIMoveTimes"]["average"],
                                                                         dict2[type]["AIMoveTimes"]["s"],
                                                                         dict2[type]["AIMoveTimes"]["n"])
        print(mean_diff, ci, p_value, t_stat1, interpretation1)

    
if __name__ == "__main__":
    # run the stat code based on the json file
    results = load_json()
    version_list = list(results.keys())
    for i, version in enumerate(version_list):
        if len(version_list) < 2:
            break
        if i == len(version_list) - 1:
            continue
        compare_versions(results[version_list[i]], results[version_list[i + 1]])