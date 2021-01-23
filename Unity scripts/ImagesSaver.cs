using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImagesSaver : MonoBehaviour
{
    // Start is called before the first frame update
    public int capturedFrames;
    int frame; 
    ImageSynthesis sintetizer; 
    public Controller controller; 

    void Start()
    {
     frame = 0; 
     sintetizer = GetComponent<ImageSynthesis>();
     Screen.SetResolution(640,480, true);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        frame ++; 
        sintetizer.Save(frame, Screen.width, Screen.height);
        if(frame == capturedFrames){
            AppHelper.Quit();
        }

    }
}
