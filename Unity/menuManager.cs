// Para cambiar de escenas, es un script muy sencillo, aquí lo importante es el concepto de escenas: 

// using UnityEngine;
// using UnityEngine.SceneManagement;
public class MenuManager : MonoBehaviour
{
public void ChangeScene(int idScene)
{
SceneManager.LoadScene(idScene);
}

public void ExitGame()
{
Application.Quit();
Debug.Log("Quit!");
}
}