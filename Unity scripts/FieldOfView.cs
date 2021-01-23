using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
public class FieldOfView : MonoBehaviour 
{

    public float viewRadius;
    public float viewAngle;
    public LayerMask targetMask;
    public LayerMask obstacleMask;
    public List<Transform> visibleTargets = new List<Transform>();
    public Transform closestTarget = null;
    public Controller controller;
    public List<Transform> closeObstacles = new List<Transform>();
    public Transform closestObstacle = null;

    void Start()
    {
        if(true){
            StartCoroutine("FindTargetsWithDelay", .01f);    
        }
    }

    IEnumerator FindTargetsWithDelay(float delay)
    {
        while (true)
        {
            yield return new WaitForSeconds(delay);
            FindVisibleTargets();
            FindCloseObstacles();
            FindClosestTarget();
            FindClosestObstacle();
            if(visibleTargets.Count == 0 ){
                closestTarget = null;
            }
            if(closeObstacles.Count == 0){
                closestObstacle = null;
            }
        }
    }

    void FindCloseObstacles()
    {
        closeObstacles.Clear();
        Collider[] closeObstaclesInViewRadius = Physics.OverlapSphere(transform.position, viewRadius, obstacleMask);

        for (int i = 0; i < closeObstaclesInViewRadius.Length ; i++)
        {
            Transform target = closeObstaclesInViewRadius[i].transform;
            Vector3 dirToTarget = (target.position - transform.position).normalized;
            float dstToTarget = Vector3.Distance(transform.position, target.position);
            closeObstacles.Add(target);
        }
    }

    void FindVisibleTargets()
    {
        visibleTargets.Clear();
        Collider[] targetsInViewRadius = Physics.OverlapSphere(transform.position, viewRadius, targetMask);

        for (int i = 0; i < targetsInViewRadius.Length; i++)
        {
            Transform target = targetsInViewRadius[i].transform;
            Vector3 dirToTarget = (target.position - transform.position).normalized;
            if (Vector3.Angle(transform.forward, dirToTarget) < viewAngle / 2)
            {
                float dstToTarget = Vector3.Distance(transform.position, target.position);
                visibleTargets.Add(target);
            }
        }
    }
    public Vector3 DirFromAngle(float angleInDegrees, bool angleIsGlobal)
    {
        if (!angleIsGlobal)
        {
            angleInDegrees += transform.eulerAngles.y;
        }
        return new Vector3(Mathf.Sin(angleInDegrees * Mathf.Deg2Rad), 0, Mathf.Cos(angleInDegrees * Mathf.Deg2Rad));
    }

    public void FindClosestTarget()
    {
        if (visibleTargets.Count != 0)
        {
            closestTarget = visibleTargets[0];
            foreach (Transform target in visibleTargets)
            {
                float distNew = Vector3.Distance(transform.position, target.position);
                float distClose = Vector3.Distance(transform.position, closestTarget.position);
                if (distNew < distClose)
                {
                    closestTarget = target;
                }
            }
        }
    }

    public void FindClosestObstacle()
    {
        if (closeObstacles.Count != 0)
        {
            closestObstacle = closeObstacles[0];
            foreach (Transform target in closeObstacles)
            {
                float distNew = Vector3.Distance(transform.position, target.position);
                float distClose = Vector3.Distance(transform.position, closestObstacle.position);
                if (distNew < distClose)
                {
                    closestObstacle = target;
                }
            }
        }
    }

}
