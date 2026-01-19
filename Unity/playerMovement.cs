
// Aquí os pongo el código de C#

// Es importante que además de las funciones Start() y Update(), tenemos la función FixedUpdate() para las físicas de los objetos:

// using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
public int playerID;
public float playerVelocity;

private Rigidbody2D _rb;
private Vector2 _playerDirection;
void Start()
{
_rb = GetComponent<Rigidbody2D>();
}
void Update()
{
float directionYp1 = Input.GetAxisRaw("Vertical");
float directionYp2 = Input.GetAxisRaw("Vertical2");

if (playerID == 1)
{
_playerDirection = new Vector2(0, directionYp1).normalized;
}
else if (playerID == 2)
{
_playerDirection = new Vector2(0, directionYp2).normalized;
}
}

private void FixedUpdate()
{
_rb.velocity = _playerDirection * playerVelocity;
}
}