// BallMovement.cs, aquí va el código:

// using System.Collections;
// using UnityEngine;

public class BallMovement : MonoBehaviour
{
public ScoreManager scoreManager;
public GameObject hitSfx;
public float startSpeed;
public float extraSpeed;
public float maxSpeed;

private int _hitCounter;
private bool _player1Start = true;
private Rigidbody2D _rb;

private void Start()
{
_rb = GetComponent<Rigidbody2D>();
StartCoroutine(Launch());
}

private IEnumerator Launch()
{
RestartBall();
_hitCounter = 0;
yield return new WaitForSeconds(1);

MoveBall(_player1Start ? new Vector2(-1, 0) : new Vector2(1, 0));
}

private void MoveBall(Vector2 direction)
{
direction = direction.normalized;

float ballSpeed = startSpeed + (_hitCounter * extraSpeed);

_rb.velocity = direction * ballSpeed;
}

private void IncreaseHitCounter()
{
if (_hitCounter * extraSpeed < maxSpeed)
_hitCounter++;
}

private void RestartBall()
{
_rb.velocity = new Vector2(0, 0);
transform.position = new Vector2(0, 0);
}

private void Bounce(Collision2D collision2D)
{
Vector3 ballPosition = transform.position;
Vector3 playerPosition = collision2D.transform.position;
float playerHeight = collision2D.collider.bounds.size.y;

float positionX;

if (collision2D.gameObject.CompareTag("Player1"))
positionX = 1;
else
positionX = -1;

float positionY = (ballPosition.y - playerPosition.y) / playerHeight;

IncreaseHitCounter();
MoveBall(new Vector2(positionX, positionY));
}

private void OnCollisionEnter2D(Collision2D col)
{
if (col.gameObject.CompareTag("Player1") || col.gameObject.CompareTag("Player2"))
Bounce(col);
else if (col.gameObject.CompareTag("RightBorder"))
{
scoreManager.Player1Goal();
_player1Start = false;
StartCoroutine(Launch());
}
else if (col.gameObject.CompareTag("LeftBorder"))
{
scoreManager.Player2Goal();
_player1Start = true;
StartCoroutine(Launch());
}

GameObject tmpHit = Instantiate(hitSfx, transform.position, transform.rotation);
Destroy(tmpHit, 1);
}
}