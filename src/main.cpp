/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File main.cpp
* Description: dvpp sample main func
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <thread>
#include <future>
#include <chrono>
#include <queue>

#include "object_detect.h"
#include "classify_process.h"
#include "face_detect.h"
#include "model_process.h"
#include "utils.h"

using namespace std::chrono;
using namespace std;

namespace {
const uint32_t kModelWidth1 = 416;
const uint32_t kModelHeight1 = 416;
const char* kModelPath1 = "../model/yolov3.om";

const uint32_t kModelWidth2 = 224;
const uint32_t kModelHeight2 = 224;
const char* kModelPath2 = "../model/googlenet.om";

const uint32_t kModelWidth3 = 304;
const uint32_t kModelHeight3 = 300;
const char* kModelPath3 = "../model/face_detection.om";
}

int32_t deviceId = 0;
string videoFile = "../data/person.mp4";
queue<cv::Mat> q1, q2, q3;

Result InitResource() {
    // ACL init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Acl init failed");
        return FAILED;
    }
    INFO_LOG("Acl init success");

//    uint32_t devcnt;
//    ret = aclrtGetDeviceCount(&devcnt);
//    if (ret != ACL_ERROR_NONE) {
//        ERROR_LOG("Acl get device count failed");
//        return FAILED;
//    }
//    INFO_LOG("Get Device Count:%d", devcnt);

    // open device
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Acl open device %d failed", deviceId);
        return FAILED;
    }
    INFO_LOG("Open device %d success", deviceId);
    //获取当前应用程序运行在host还是device
    aclrtRunMode runMode_;
    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    INFO_LOG("Get Run Mode success");

    return SUCCESS;
}

Result ReleaseResource() {
    aclError ret;
    ret = aclrtResetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
    return SUCCESS;
}

Result face_detection(string channel_name, aclrtContext context, aclrtStream stream) {
    //显式创建一个Context，用于管理Stream对象。
    aclError ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    //显式创建一个Stream
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    FaceDetect face_detect(kModelPath3, kModelWidth3, kModelHeight3);
    //Initializes the ACL resource for categorical reasoning, loads the model and requests the memory used for reasoning input
    Result ret1 = face_detect.Init(channel_name);
    if (ret1 != SUCCESS) {
        ERROR_LOG("FaceDetection Init resource failed");
        return FAILED;
    }
    //Frame by frame reasoning
    while (!q3.empty()) {
        //从队列里获取每一帧图片
        cv::Mat frame = q3.front();
        q3.pop();
        //对帧图片进行预处理
        high_resolution_clock::time_point start = high_resolution_clock::now();
        Result ret = face_detect.Preprocess(frame);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
            videoFile.c_str());
            continue;
        }
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration < double, std::milli > time_span = end - start;
        cout << "\nPreProcess time " << time_span.count() << "ms" << endl;

        //将预处理的图片送入模型推理,并获取推理结果
        aclmdlDataset* inferenceOutput = nullptr;
        start = high_resolution_clock::now();
        ret = face_detect.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "Inference time " << time_span.count() << "ms" << endl;

        //解析推理输出,并将推理得到的物体类别,置信度和图片送到presenter server显示
        start = high_resolution_clock::now();
        ret = face_detect.Postprocess(frame, inferenceOutput);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "PostProcess time " << time_span.count() << "ms" << endl;
    }

    INFO_LOG("Execute video face detection success");

    //释放资源
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);

    return SUCCESS;
}

Result object_detection(string channel_name, aclrtContext context, aclrtStream stream) {
    //显式创建一个Context，用于管理Stream对象。
    aclError ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    //显式创建一个Stream
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    ObjectDetect detect(kModelPath1, kModelWidth1, kModelHeight1);
    //Initializes the ACL resource for categorical reasoning, loads the model and requests the memory used for reasoning input
    Result ret1 = detect.Init(channel_name);
    if (ret1 != SUCCESS) {
        ERROR_LOG("ObjectDetection Init resource failed");
        return FAILED;
    }
    //Frame by frame reasoning
    while (!q1.empty()) {
        //从队列里获取每一帧图片
        cv::Mat frame = q1.front();
        q1.pop();
        //对帧图片进行预处理
        high_resolution_clock::time_point start = high_resolution_clock::now();
        Result ret = detect.Preprocess(frame);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
            videoFile.c_str());
            continue;
        }
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration < double, std::milli > time_span = end - start;
        cout << "\nPreProcess time " << time_span.count() << "ms" << endl;

        //将预处理的图片送入模型推理,并获取推理结果
        aclmdlDataset* inferenceOutput = nullptr;
        start = high_resolution_clock::now();
        ret = detect.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "Inference time " << time_span.count() << "ms" << endl;

        //解析推理输出,并将推理得到的物体类别,置信度和图片送到presenter server显示
        start = high_resolution_clock::now();
        ret = detect.Postprocess(frame, inferenceOutput);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "PostProcess time " << time_span.count() << "ms" << endl;
    }

    INFO_LOG("Execute video object detection success");

    //释放资源
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);

    return SUCCESS;
}

Result classify_process(string channel_name, aclrtContext context, aclrtStream stream) {
    //显式创建一个Context，用于管理Stream对象。
    aclError ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    //显式创建一个Stream
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    ClassifyProcess classify(kModelPath2, kModelWidth2, kModelHeight2);
    Result ret2 = classify.Init(channel_name);
    if (ret2 != SUCCESS) {
        ERROR_LOG("Classification Init resource failed");
        return FAILED;
    }
    //逐帧推理
    while (!q2.empty()) {
        //从队列里获取每一帧图片
        cv::Mat frame = q2.front();
        q2.pop();
        //对帧图片进行预处理
        high_resolution_clock::time_point start = high_resolution_clock::now();
        Result ret = classify.Preprocess(frame);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
            videoFile.c_str());
            continue;
        }
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration < double, std::milli > time_span = end - start;
        cout << "\nPreProcess time " << time_span.count() << "ms" << endl;

        //将预处理的图片送入模型推理,并获取推理结果
        aclmdlDataset* inferenceOutput = nullptr;
        start = high_resolution_clock::now();
        ret = classify.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "Inference time " << time_span.count() << "ms" << endl;

        //解析推理输出,并将推理得到的物体类别,置信度和图片送到presenter server显示
        start = high_resolution_clock::now();
        ret = classify.Postprocess(frame, inferenceOutput);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "PostProcess time " << time_span.count() << "ms" << endl;
    }

    INFO_LOG("Execute video object detection success");

    //释放资源（context stream）
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);

    return SUCCESS;
}

void read_frame(cv::VideoCapture capture){
    while (1) {
        cv::Mat frame;
        if (!capture.read(frame)) {
            INFO_LOG("Video capture return false");
            break;
        }
//        if( q1.size() == 1)
//            break;
//        else
//            q1.push(frame);
//
//        if( q2.size() == 1)
//            break;
//        else
//            q2.push(frame);
//
//        if( q3.size() == 1)
//            break;
//        else
//            q3.push(frame);

        if( q1.size() <= 10){
            q1.push(frame);
        }

        if( q2.size() <= 10){
            q2.push(frame);
        }

        if( q3.size() <= 10){
            q3.push(frame);
        }
//        sleep(0.1);
    }
}

int main(int argc, char *argv[]) {
    //检查应用程序执行时的输入,程序执行的参数为输入视频文件路径
    if((argc < 2) || (argv[1] == nullptr)){
        ERROR_LOG("Please input: ./main <image_dir>");
        return FAILED;
    }

//    使用opencv打开视频流
    videoFile = string(argv[1]);
    printf("open %s\n", videoFile.c_str());
    cv::VideoCapture capture(videoFile);
    if (!capture.isOpened()) {
        cout << "Movie open Error" << endl;
        return FAILED;
    }

    //获取视频流信息（分别率、fps等）
    cout << "width = " << capture.get(3) << endl;
    cout << "height = " << capture.get(4) << endl;
    cout << "frame_fps = " << capture.get(5) << endl;
    cout << "frame_nums = " << capture.get(7) << endl;

    //创建一个分支线程,持续不断的读视频流
    std::thread t(read_frame, capture);
//    sleep(1);

    //初始化acl资源
    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    aclrtContext context;
    aclrtStream stream;
    //分别创建线程去执行不同的推理
    std::thread t1(object_detection, "object_detection_video", context, stream);
    std::thread t2(classify_process, "classification_video", context, stream);
//    sleep(1);
    std::thread t3(face_detection, "face_detection_video", context, stream);
    t.join();
    t1.join();
    t2.join();
    t3.join();

    //释放创建的acl资源
    ReleaseResource();

}
