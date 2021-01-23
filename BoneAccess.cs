using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BoneAccess : MonoBehaviour
{
    public Vector3 Offset; 
    public Transform Target; 

    Animator animator; 
    Transform head; 
    // Start is called before the first frame update
    void Start()
    {
        animator = GetComponent<Animator>();
        head = GetComponent<Transform>().Find("Anna/hips/spine/chest/chest1/neck/head");
    }

    // Update is called once per frame
    void LateUpdate()
    {   
        if(Target.gameObject.activeSelf){
            head.LookAt(Target.position);
            var dir = head.position - this.transform.position; 
            var angle =  Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

        }
    }

}