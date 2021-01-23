using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;

public class Controller : MonoBehaviour
{
    public float moveSpeed = 6;

    //Model Parmas
    public bool Male;
    public bool Female;
    public string modelNumber;
    public string seqNumber;
    int framecount;
    string pathTxt;
    string pathJson;

    public bool repeating;
    SavingPoint targetSavingPoint;
    int currentSavingPoint;

    Rigidbody rigidbody;
    Camera viewCamera;
    Vector3 velocity;
    public StartParams SetUp;
    FieldOfView fow;
    public Transform head;
    public float focusTime;

    Vector3 originalEuler; 
    Vector3 angles;
    List<SavingPoint> savPoints = new List<SavingPoint>();
    SavingPoint[] Spoints;
    Dictionary<int, SavingPoint> dictionaryS = new Dictionary<int, SavingPoint>();

    public bool capture = false;

    //WayPoints
    public Transform[] wayPointsList;
    public int currentWayPoint = 0;
    Transform targetWayPoint;
    public Animator animator;
    public float area; 
    public GameObject wayPoint; 

    //Random Trajectory
    public float rotationSpeed; 
    bool focus; 
    bool validPoint;
    Vector3 checkPoint;

    //Follow trajectoreis
    Quaternion fixHead; 
    TrajectoryPoint[] points;
    Dictionary<int, TrajectoryPoint> dictionary = new Dictionary<int, TrajectoryPoint>();
    public Vector2 imageSize = new Vector2(640,480);
    TrajectoryPoint targetTrajPoint;
    int currentTrajPoint = 0;
    public float bodyRot = 0f; 
    public float FOV = 80f;

    void Start()
    {
        rigidbody = GetComponent<Rigidbody>();
        viewCamera = Camera.main;
        string json = File.ReadAllText(Application.dataPath + "/Config.json");
        SetUp = JsonUtility.FromJson<StartParams>(json);    

        transform.position = SetUp.position;
        head = GetComponentInChildren<Transform>().Find("Unity compliant skeleton/hips/spine/chest/chest1/neck/head");
        fixHead = head.rotation;
        originalEuler = GetPitchYawRollDeg(Quaternion.LookRotation(head.transform.forward));

        fow = GetComponentInChildren<FieldOfView>();
        focus = false;
        
        /*
        string path;
        if(!repeating){
            pathJson = SetUpFile("json");
        }else{
            string jsonTraj = File.ReadAllText("C:/TFG/DATASET/Male models/1/1_1.json");
            Spoints = JsonHelper.FromJson<SavingPoint>(jsonTraj); 
            transform.forward = new Vector3(Spoints[0].rotx, Spoints[0].roty, Spoints[0].rotz);
        }*/

        if(SetUp.randomTrajectory && !repeating){
            int select = Random.Range(1,4);
            transform.rotation =  Quaternion.Euler(new Vector3(0, Random.Range(0, 360), 0));
            var point = Random.insideUnitCircle * area;
            wayPoint = new GameObject("wayPoint");
            var direction = new Vector3(point.x,transform.position.y,point.y);
            wayPoint.transform.position = transform.position + direction;
            pathTxt = SetUpFile("txt");
        } 
        
        if(SetUp.followRealTrajectories){
            string jsonTraj = File.ReadAllText(Application.dataPath + "/Path.json");
            points = JsonHelper.FromJson<TrajectoryPoint>(jsonTraj); 
            foreach(TrajectoryPoint point in points){
                dictionary.Add(point.frame, point);
            }
        }

        
    }

    // Update is called once per frame
/*
    void FixedUpdate(){
        framecount++;
        angles = GetPitchYawRollDeg(Quaternion.LookRotation(head.transform.forward));
        angles = new Vector3(angles.x, angles.y, angles.z);
        capture = true;
        string content ="   {Frame:" + framecount + " Point: " + UnitsToPixels(head.transform.position) + " Pitch: " + angles.x + " Yaw: " + angles.y + " Roll: " + angles.z + "\n";
        File.AppendAllText(pathTxt, content);
        Debug.Log(content);
        capture = false;
    }
    */
    void LateUpdate()
    {
        /*
        //FOLLOW MOUSE
            Vector3 mousePos = viewCamera.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, viewCamera.transform.position.y));
            transform.LookAt(mousePos + Vector3.up * transform.position.y);
            velocity = new Vector3(Input.GetAxisRaw("Horizontal"), 0, Input.GetAxisRaw("Vertical")).normalized * moveSpeed;
        */

        if(SetUp.followRealTrajectories){
           head.rotation = fixHead;
            //PROVES 477, 569 
            
            TrajectoryPoint point;
            if(dictionary.TryGetValue(477, out point)){
                RecreatePoint(point);
            } /*
            if (currentTrajPoint < points.Length )
            {
                if (targetTrajPoint == null) targetTrajPoint = points[currentTrajPoint];
                followTrajectory(); 
            }  */

        } else if (repeating){
            //head.rotation = fixHead;            
            if (currentSavingPoint < Spoints.Length )
            {
                if (targetSavingPoint == null) targetSavingPoint = Spoints[currentSavingPoint];

                if(fow.closestTarget != null){
                    GetComponent<Animator>().enabled = false; 
                    look(fow.closestTarget);
                    focus = true;
                    focusTime = Random.Range(1f, 4f);
                    StartCoroutine("Wait", focusTime);
                }

                if(!focus){
                    followSavedTrajectory();

                }


            } //if(currentSavingPoint == Spoints.Length) //AppHelper.Quit();
            
        } else {
            if (currentWayPoint < this.wayPointsList.Length ) {
                if (targetWayPoint == null) targetWayPoint = wayPointsList[currentWayPoint];
                

                if(fow.closestTarget != null){
                    GetComponent<Animator>().enabled = false; 
                    look(fow.closestTarget);
                    focus = true;
                   // focusTime = 2f;//Random.Range(1f, 4f);
                    StartCoroutine("Wait", focusTime);

                } 

                if(!focus){
                    if(SetUp.randomTrajectory){
                        WalkRandom();
                    } else {
                        FollowWayPoints();
                    }
                }

                
                
            }
            /*
            if(framecount <=500 ){
                SavingPoint aux = new SavingPoint(); 
                aux.frame = framecount;
                aux.x = transform.position.x; 
                aux.y = transform.position.y;
                aux.z = transform.position.z;
                aux.rotx = transform.forward.x;
                aux.roty = transform.forward.y;
                aux.rotz = transform.forward.z;                    
                savPoints.Add(aux);
            }
            if(framecount == 500){
                string TraJson = JsonHelper.ToJson(savPoints.ToArray(),true);
                Debug.Log(TraJson);
                File.AppendAllText(pathJson, TraJson);
                AppHelper.Quit();
            } */
        }
    }

    Vector3 UnitsToPixels(Vector3 position){
        var h = viewCamera.pixelHeight;
        var w = viewCamera.pixelWidth;
        
        var newPoint = viewCamera.WorldToScreenPoint(position);

        var newX = newPoint.x * imageSize.x / w;
        var newY = newPoint.y * imageSize.y / h;

        var result = new Vector3(newX, imageSize.y - newY, newPoint.z);

        
        return result;
    }

    Vector3 PixelsToUnits(TrajectoryPoint point){
        var w = viewCamera.pixelHeight;
        var h = viewCamera.pixelWidth;
        //Debug.Log("CAM WIDTH: " + w + "   CAM HEIGHT" + h);

        //READ FRAME INFO
        var pixels = new Vector3(point.x, point.y);
        //Debug.Log("FRAME: " + point.frame + "    PIXEL POS: " + pixels);

        //FRAMEPIXELS TO CAMPIXELS
        var newX = point.x * viewCamera.pixelWidth / imageSize.x; 
        var newY = point.y * viewCamera.pixelHeight / imageSize.y;
        //Debug.Log("NEW PIXELS " + newX + "  " + newY);

        //TRANSFORM PIXELS TO UNITs
        var newpoint = viewCamera.ScreenToWorldPoint(new Vector3(newX, newY, 3));
        //Debug.Log("NEW" + newpoint);
        var targetUnits = new Vector3(newpoint.x, transform.position.y, newpoint.z);
        return targetUnits;
    }


    string SetUpFile(string type){
        string path = "C:/TFG/DATASET";
        if(Female){ 
            path = path + "/Female models" +"/" + modelNumber +"/" + modelNumber + "_" + seqNumber + "."+ type;
        }else{
            path = path + "/Male_models" +"/" + modelNumber + "/" + modelNumber +"_" + seqNumber + "."+ type;
        }
        if(!File.Exists(path)) {
            File.WriteAllText(path, "");
        }

        return path;
    }

    void RecreatePoint(TrajectoryPoint point){
        //CAM PIXELS
        var w = viewCamera.pixelHeight;
        var h = viewCamera.pixelWidth;
        Debug.Log("CAM WIDTH: " + w + "   CAM HEIGHT" + h);

        //READ FRAME INFO
        var origin = new Vector3(0,0);
        var pixels = new Vector3(point.x, point.y);
        Debug.Log("FRAME: " + point.frame + "    PIXEL POS: " + pixels);

        //FRAMEPIXELS TO CAMPIXELS
        var newX = point.x * viewCamera.pixelWidth / imageSize.x; 
        var newY = point.y * viewCamera.pixelHeight / imageSize.y;
        Debug.Log("NEW PIXELS " + newX + "  " + newY);

        //TRANSFORM PIXELS TO UNITs
        var newpoint = viewCamera.ScreenToWorldPoint(new Vector3(newX, newY, 3));
        Debug.Log("NEW" + newpoint);
        var targetUnits = new Vector3(newpoint.x, transform.position.y, newpoint.z);
        transform.LookAt(targetUnits);
        transform.position = targetUnits;
        //Vector3.RotateTowards(transform.forward, targetUnits - transform.position, moveSpeed * Time.deltaTime, 0.0f );

        /*
        var unitsOG = converter.ConvertToWorldUnits(origin);    
        var units = converter.ConvertToWorldUnits(pixels);
        transform.position = new Vector3(units.x, transform.position.y, units.y);*/

        head.transform.rotation = Quaternion.LookRotation(head.transform.forward)  * Quaternion.Euler(point.Pitch *  Mathf.Rad2Deg, point.Yaw * Mathf.Rad2Deg, point.Roll * Mathf.Rad2Deg);
        float headRot = head.transform.rotation.eulerAngles.y;
        float delta = headRot - bodyRot; 
        if(delta > 180){
            delta -= 360;
        }
        if(delta < -180){
            delta += 360;
        }
        if(Mathf.Abs(delta) > FOV ){
            if((delta > FOV || delta <-180) && delta < 180){
                bodyRot = headRot - FOV;
            }
            delta = headRot - bodyRot;
            if((delta < FOV || delta > 180)){
                bodyRot = headRot + FOV;
            }
        }
        Debug.Log("Head: " +  headRot + "  Body:" + bodyRot);
        transform.rotation = Quaternion.Euler(0,headRot,0);
    }
    IEnumerator Wait(float time){
        yield return new WaitForSeconds(time);
        focus = false;
        GetComponent<Animator>().enabled = true;
        fow.closestTarget = null;
    }

    void look(Transform target){
        if(fow.visibleTargets.Count == 0){
            if(SetUp.randomTrajectory){
                transform.forward = Vector3.RotateTowards(transform.forward, wayPoint.transform.position - transform.position, moveSpeed*Time.deltaTime, 0.0f);
            }else {
                transform.forward = Vector3.RotateTowards(transform.forward, targetWayPoint.position - transform.position, moveSpeed*Time.deltaTime, 0.0f);
            }
        }else{
            head.transform.forward = Vector3.RotateTowards(head.transform.forward, target.position - head.transform.position,  moveSpeed * Time.deltaTime, 0.0f );
         
            //Debug.Log("after:" + head.transform.forward + "Pitch: " + angles.x + "Yaw: " + angles.y);

        }   
    }

    void followSavedTrajectory(){
        var point = new Vector3(targetSavingPoint.x, targetSavingPoint.y, targetSavingPoint.z);
        transform.position = Vector3.MoveTowards(transform.position, point, moveSpeed * Time.deltaTime);
        if(transform.position == point){
            currentSavingPoint++;
            targetSavingPoint = Spoints[currentSavingPoint];
        }
    }

    void followTrajectory(){
        var point = PixelsToUnits(targetTrajPoint);
        transform.position = Vector3.MoveTowards(transform.position, point, moveSpeed * Time.deltaTime);
        if(transform.position == point){
            currentTrajPoint++;
            targetTrajPoint = points[currentTrajPoint];
        }

        head.transform.rotation = Quaternion.LookRotation(head.transform.forward) * Quaternion.Euler(targetTrajPoint.Pitch *  Mathf.Rad2Deg, targetTrajPoint.Yaw * Mathf.Rad2Deg, targetTrajPoint.Roll * Mathf.Rad2Deg);
        var aux = new Vector3(targetTrajPoint.Pitch *  Mathf.Rad2Deg, targetTrajPoint.Yaw * Mathf.Rad2Deg, targetTrajPoint.Roll * Mathf.Rad2Deg);

        float headRot = head.transform.rotation.eulerAngles.y;
        float delta = headRot - bodyRot; 
        if(delta > 180){
            delta -= 360;
        }
        if(delta < -180){
            delta += 360;
        }
        if(Mathf.Abs(delta) > FOV ){
            if((delta > FOV || delta <-180) && delta < 180){
                bodyRot = headRot - FOV;
            }
            delta = headRot - bodyRot;
            if((delta < FOV || delta > 180)){
                bodyRot = headRot + FOV;
            }
        }
        Debug.Log("Head: " +  headRot + "  Body:" + bodyRot);
        transform.rotation = Quaternion.Euler(0,bodyRot,0);
    }

    void FollowWayPoints(){
        transform.position = Vector3.MoveTowards(transform.position, targetWayPoint.position, moveSpeed * Time.deltaTime);
        if (transform.position == targetWayPoint.position)
        {
            currentWayPoint++;
            targetWayPoint = wayPointsList[currentWayPoint];
        }
        transform.forward = Vector3.RotateTowards(transform.forward, wayPoint.transform.position - transform.position, 2*moveSpeed*Time.deltaTime, 0.0f);

    }

    void WalkRandom(){
        transform.position = Vector3.MoveTowards(transform.position, wayPoint.transform.position, moveSpeed * Time.deltaTime);
        if(transform.position == wayPoint.transform.position)
        {   
            RaycastHit hit;
            var point = Random.insideUnitCircle * area;
            if((point.x <= 1.01 || point.x >= 5.57)||(point.y <= -1.98 || point.y >= 1.12)){
                point = Random.insideUnitCircle * area;
               // Debug.Log("fixed to: " + point);
            }
            var oldPoint = wayPoint.transform.position;
            var direction= new Vector3(point.x,oldPoint.y,point.y); 
            wayPoint.transform.position = oldPoint + direction;


            if(fow.closestObstacle != null){
                if(Physics.Raycast(transform.position, direction,out hit, fow.viewRadius, fow.obstacleMask)){
                    Debug.Log("Collision evaded with " + hit.transform.gameObject.name);
                    var newDirection = new Vector3(-direction.x, direction.y, -direction.x);
                    wayPoint.transform.position = transform.position + newDirection;
                }
            }
        }



        transform.forward = Vector3.RotateTowards(transform.forward, wayPoint.transform.position - transform.position, 2*moveSpeed*Time.deltaTime, 0.0f);
    }

    public static Vector3 GetPitchYawRollRad(Quaternion q)
    {
        /*
        float roll = Mathf.Atan2(2*rotation.y*rotation.w - 2*rotation.x*rotation.z, 1 - 2*rotation.y*rotation.y - 2*rotation.z*rotation.z);
        float pitch = Mathf.Atan2(2*rotation.x*rotation.w - 2*rotation.y*rotation.z, 1 - 2*rotation.x*rotation.x - 2*rotation.z*rotation.z);
        float yaw = Mathf.Asin(2*rotation.x*rotation.y + 2*rotation.z*rotation.w);
        */
        float pitch;
        float yaw;
        float roll;

        // roll (x-axis rotation)
        float sinr_cosp = 2 * (q.w * q.x + q.z * q.y);
        float cosr_cosp = 1 - 2 * (q.x * q.x + q.z * q.z);
        roll = Mathf.Atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        float sinp = 2 * (q.w * q.z - q.y * q.x);
        if (Mathf.Abs(sinp) >= 1)
            pitch = Mathf.PI / 2 * Mathf.Sign(sinp); // use 90 degrees if out of range
        else
            pitch = Mathf.Asin(sinp);

        // yaw (z-axis rotation)
        float siny_cosp = 2 * (q.w * q.y + q.x * q.z);
        float cosy_cosp = 1 - 2 * (q.z * q.z + q.y * q.y);
        yaw = Mathf.Atan2(siny_cosp, cosy_cosp);
                 
        return new Vector3(pitch, yaw, roll);
    }
             
    public static Vector3 GetPitchYawRollDeg(Quaternion rotation)
    {
        Vector3 radResult = GetPitchYawRollRad(rotation);
        return new Vector3(radResult.x * Mathf.Rad2Deg, radResult.y * Mathf.Rad2Deg, radResult.z * Mathf.Rad2Deg);
    }

    public Vector3 StartRotation(int selector){
        if(selector == 1){
            return transform.forward; 
        } 
        if(selector == 2){
            return -transform.forward;
        }
        if(selector == 3){
            return transform.right;
        }
        if(selector == 4){
            return -transform.right;
        }
        return transform.forward;
    }

}


public class StartParams
{
    public Vector3 position;
    public float headPitch;
    public float headYaw;
    public bool randomTrajectory;
    public bool followRealTrajectories;

}

[System.Serializable]
public class TrajectoryPoint    {
    public int frame;
    public int x; 
    public int y; 
    public int z;
    public float Yaw; 
    public float Pitch;
    public float Roll;
}

[System.Serializable]
public class SavingPoint   {
    public int frame;
    public float x; 
    public float y; 
    public float z;
    public float rotx; 
    public float roty;
    public float rotz;

}

public static class JsonHelper
{
    public static T[] FromJson<T>(string json)
    {
        Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
        return wrapper.Items;
    }

    public static string ToJson<T>(T[] array)
    {
        Wrapper<T> wrapper = new Wrapper<T>();
        wrapper.Items = array;
        return JsonUtility.ToJson(wrapper);
    }

    public static string ToJson<T>(T[] array, bool prettyPrint)
    {
        Wrapper<T> wrapper = new Wrapper<T>();
        wrapper.Items = array;
        return JsonUtility.ToJson(wrapper, prettyPrint);
    }

    private class Wrapper<T>
    {
        public T[] Items;
    } 
}