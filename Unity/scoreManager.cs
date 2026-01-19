
// Sería el código C# para el MARCADOR (SCORE):

// using UnityEngine;
// using UnityEngine.SceneManagement;
// using TMPro;

public class ScoreManager : MonoBehaviour
{
public int maxScore;
public TextMeshProUGUI player1ScoreText, player2ScoreText;

private int _player1Score, _player2Score;

public void Player1Goal()
{
_player1Score++;
player1ScoreText.text = _player1Score.ToString();
CheckScore();
}
public void Player2Goal()
{
_player2Score++;
player2ScoreText.text = _player2Score.ToString();
CheckScore();
}

private void CheckScore()
{
if (_player1Score == maxScore || _player2Score == maxScore)
SceneManager.LoadScene(2);
}
}
