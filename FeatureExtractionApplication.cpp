// FeatureExtractionApplication.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "stdafx.h"
#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2\highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <windows.h>

#define FREQ_COUNT 44
#define POSITION_COUNT 25
#define DYNAMIC_FEATURE_COUNT 8
#define STATIC_FEATURE_COUNT 5
#define FEATURE_COUNT DYNAMIC_FEATURE_COUNT+STATIC_FEATURE_COUNT
#define DEVISE 0
#define PI 3.1415926535

using namespace std;
using namespace cv;

float evaluate_angle(Point3f c, Point3f a, Point3f b, int freq, int joint);
float evaluate_seperated_angle(Point3f pA, Point3f pB, Point3f pC, Point3f pD);
float evaluate_dist(Point3f a, Point3f b);
float calc_tall(vector<vector<Point3f>> v);
float calc_length(vector<vector<Point3f>> v, vector<int> joints);
float calc_shoulderLen(vector<vector<Point3f>> v, vector<int> joints);
float calc_armLeftLen(vector<vector<Point3f>> v, vector<int> joints);
float calc_armRightLen(vector<vector<Point3f>> v, vector<int> joints);
float calc_bodyLen(vector<vector<Point3f>> v, vector<int> joints);

enum JointType
{
	JointType_SpineBase = 0,
	JointType_SpineMid = 1,
	JointType_Neck = 2,
	JointType_Head = 3,
	JointType_ShoulderLeft = 4,
	JointType_ElbowLeft = 5,
	JointType_WristLeft = 6,
	JointType_HandLeft = 7,
	JointType_ShoulderRight = 8,
	JointType_ElbowRight = 9,
	JointType_WristRight = 10,
	JointType_HandRight = 11,
	JointType_HipLeft = 12,
	JointType_KneeLeft = 13,
	JointType_AnkleLeft = 14,
	JointType_FootLeft = 15,
	JointType_HipRight = 16,
	JointType_KneeRight = 17,
	JointType_AnkleRight = 18,
	JointType_FootRight = 19,
	JointType_SpineShoulder = 20,
	JointType_HandTipLeft = 21,
	JointType_ThumbLeft = 22,
	JointType_HandTipRight = 23,
	JointType_ThumbRight = 24,
};

enum StaticFeatureType{
	Feature_ShoulderLength = 0,
	Feature_Tall = 1,
	Feature_ArmLeftLen = 2,
	Feature_ArmRightLen = 3,
	Feature_BodyLen = 4
};

enum DynamicFeatureType{
	Feature_Neck = 0,
	Feature_LeftShoulder = 1,
	Feature_RightShoulder = 2,
	Feature_LeftElbow = 3,
	Feature_RightElbow = 4,
	Feature_Hip = 5,
	Feature_LeftKnee = 6,
	Feature_RightKnee = 7,
};

int _tmain(int argc, _TCHAR* argv[])
{
	int i,j;

	string devise_filename;
	if (DEVISE == 0){
		devise_filename = "both_";
	}
    else if (DEVISE == 1){
		devise_filename = "devise1_";
	}
	else {
		devise_filename = "devise2_";
	}
	//めんどいからファイル名そのまま
	string input_position_filenames[POSITION_COUNT] = {
		"position_SpineBase.dat",
		"position_SpineMid.dat",
		"position_Neck.dat",
		"position_Head.dat",
		"position_ShoulderLeft.dat",
		"position_ElbowLeft.dat",
		"position_WristLeft.dat",
		"position_HandLeft.dat",
		"position_ShoulderRight.dat",
		"position_ElbowRight.dat",
		"position_WristRight.dat",
		"position_HandRight.dat",
		"position_HipLeft.dat",
		"position_KneeLeft.dat",
		"position_AnkleLeft.dat",
		"position_FootLeft.dat",
		"position_HipRight.dat",
		"position_KneeRight.dat",
		"position_AnkleRight.dat",
		"position_FootRight.dat",
		"position_SpineShoulder.dat",
		"position_HandTipLeft.dat",
		"position_ThumbLeft.dat",
		"position_HandTipRight.dat",
		"position_ThumbRight.dat"
	};

	const string static_feature_names[STATIC_FEATURE_COUNT] = {
		"shoulderLen",
		"tall",
		"armLeftLen",
		"armRightLen",
		"bodyLen"
	};

	const string dynamic_feature_names[DYNAMIC_FEATURE_COUNT] = {
		"neck",
		"leftShoulder",
		"rightShoulder",
		"leftElbow",
		"rightElbow",
		"hip",
		"leftKnee",
		"rightKnee"
	};


	//静的特徴量算出の際に用いる点の組み合わせ
	const vector<vector<int>> static_feature_use_points = {
		{ JointType_SpineShoulder, JointType_ShoulderRight, JointType_ShoulderLeft },
		{},
		{ JointType_ElbowLeft, JointType_ShoulderLeft, JointType_WristLeft }, 
		{ JointType_ElbowRight, JointType_ShoulderRight, JointType_WristRight },
		{ JointType_SpineMid, JointType_SpineShoulder, JointType_SpineBase }
	};

	//動的特徴量算出の際に用いる点の組み合わせ
	const vector<vector<int>> dynamic_feature_use_angles = {
		{ JointType_Neck, JointType_Head, JointType_SpineShoulder},
		{ JointType_ShoulderLeft, JointType_SpineShoulder, JointType_ElbowLeft },
		{ JointType_ShoulderRight, JointType_SpineShoulder, JointType_ElbowRight },
		{ JointType_ElbowLeft, JointType_ShoulderLeft, JointType_WristLeft },
		{ JointType_ElbowRight, JointType_ShoulderRight, JointType_WristRight },
		{ JointType_KneeLeft, JointType_HipLeft, JointType_KneeRight, JointType_HipRight },
		{ JointType_KneeLeft, JointType_HipLeft, JointType_AnkleLeft },
		{ JointType_KneeRight, JointType_HipRight, JointType_AnkleRight }
	};

	//周期セットごとに特徴量を計算
	for (int k = 0; k < FREQ_COUNT; k++){
		//入力ファイル名定義(out足す)
		string input_position_filename_base[POSITION_COUNT];
		for (i = 0; i < POSITION_COUNT; i++){
			input_position_filename_base[i] = to_string(k) + "_out_" + input_position_filenames[i];
		}

		//動的特徴量出力ファイルの定義
		string output_dynamic_filename = devise_filename + to_string(k) + "_output_dynamic_features.dat";

		//静的特徴量出力ファイルの定義
		string output_static_filename = devise_filename + to_string(k) + "_output_static_features.dat";

		//位置データ取り込み
		vector<vector<Point3f>> connected_positions;
		for (i = 0; i < POSITION_COUNT; i++){
			vector<Point3f> p;
			connected_positions.push_back(p);
		}
		for (i = 0; i < POSITION_COUNT; i++){
			ifstream input_datafile;
			input_datafile.open(input_position_filename_base[i]);
			if (input_datafile.fail()){
				cout << "Exception: ファイルが見つかりません。" << endl;
				cin.get();
			}
			string str;
			while (getline(input_datafile, str)){
				string tmp;
				istringstream stream(str);
				int c = 0;
				Point3f p;
				//一行読む(スペースでsplit)
				while (getline(stream, tmp, ' ')){
					int val = stoi(tmp);
					//範囲の判定(start以上end以下なら次のブロックを読む)
					//X座標
					if (c == 0){
						p.x = val;
					}
					//Y座標
					else if (c == 1){
						p.y = val;
					}
					//Z座標
					else{
						p.z = val;
						//スタックに貯める(push)
						connected_positions[i].push_back(p);
					}
					c++;
				}
			}
		}

		/***************************静的特徴量の算出****************************/
		vector<float> static_features;   //各周期に5つの特徴量
		ofstream output_static_file(output_static_filename);
		for (i = 0; i < STATIC_FEATURE_COUNT; i++){
			float dist;
			switch (i){
			case Feature_ShoulderLength:
				dist = calc_shoulderLen(connected_positions, static_feature_use_points[Feature_ShoulderLength]);
				break;
			case Feature_Tall:
				dist = calc_tall(connected_positions);
				break;
			case Feature_ArmLeftLen:
				dist = calc_armLeftLen(connected_positions, static_feature_use_points[Feature_ArmLeftLen]);
				break;
			case Feature_ArmRightLen:
				dist = calc_armRightLen(connected_positions, static_feature_use_points[Feature_ArmRightLen]);
				break;
			case Feature_BodyLen:
				dist = calc_bodyLen(connected_positions, static_feature_use_points[Feature_BodyLen]);
				break;
			default:
				break;
			}
			output_static_file << dist << endl;
		}
		output_static_file.close();

		/***************************動的特徴量の算出****************************/
		//動的特徴量を保持するためのvector
		vector<vector<float>> dynamic_features;
		for (i = 0; i < DYNAMIC_FEATURE_COUNT; i++){
			vector<float> feature;
			dynamic_features.push_back(feature);
		}

		//特徴量算出=>出力
		for (i = 0; i < DYNAMIC_FEATURE_COUNT; i++){
			vector<int> use_points = dynamic_feature_use_angles[i];
			if (use_points.size() == 3){
				vector<Point3f> p1 = connected_positions[use_points[0]];
				vector<Point3f> p2 = connected_positions[use_points[1]];
				vector<Point3f> p3 = connected_positions[use_points[2]];

				//フレームサイズを最小値に合わせてエラー回避
				int frameSize;
				int p1Size = p1.size();
				int p2Size = p2.size();
				int p3Size = p3.size();
				int min = 10000;
				if (p1Size == p2Size && p2Size == p3Size){
					frameSize = p1Size;
				}
				else {
					min = p1Size;
					frameSize = p1Size;
					if (p2Size < min){
						min = p2Size;
						frameSize = p2Size;
					}
					if (p3Size < min){
						min = p3Size;
						frameSize = p3Size;
					}
				}

				for (j = 0; j < frameSize; j++){
					float angle;
					if (p1[j].x == 0.0 && p1[j].y == 0.0 && p1[j].z == 0.0){
						angle = 0.0;
					}
					else{
						angle = evaluate_angle(p1[j], p2[j], p3[j], k, i);
					}
					dynamic_features[i].push_back(angle);
				}
			}
			else if (use_points.size() == 4){
				vector<Point3f> p1 = connected_positions[use_points[0]];
				vector<Point3f> p2 = connected_positions[use_points[1]];
				vector<Point3f> p3 = connected_positions[use_points[2]];
				vector<Point3f> p4 = connected_positions[use_points[3]];
				//フレームサイズを最小値に合わせてエラー回避
				int frameSize;
				int p1Size = p1.size();
				int p2Size = p2.size();
				int p3Size = p3.size();
				int min = 10000;
				if (p1Size == p2Size && p2Size == p3Size){
					frameSize = p1Size;
				}
				else {
					min = p1Size;
					frameSize = p1Size;
					if (p2Size < min){
						min = p2Size;
						frameSize = p2Size;
					}
					if (p3Size < min){
						min = p3Size;
						frameSize = p3Size;
					}
				}
				for (j = 0; j < frameSize; j++){
					float angle;
					if (p1[j].x == 0.0 && p1[j].y == 0.0 && p1[j].z == 0.0){
						angle = 0.0;
					}
					else{
						angle = evaluate_seperated_angle(p1[j], p2[j], p3[j], p4[j]);
					}
					dynamic_features[i].push_back(angle);
				}
			}
			else{
				cout << "予期せぬエラー" << endl;
			}
		}
		//平均値で引く=>ファイル出力
		ofstream output_dynamic_file(output_dynamic_filename);
		for (i = 0; i < DYNAMIC_FEATURE_COUNT; i++){
			string feature_filename = devise_filename + to_string(k) + "_" + dynamic_feature_names[i] + ".dat";
		//	ofstream output_feature_file(feature_filename);
			vector<float> feature = dynamic_features[i];
			for (auto itr = feature.begin(); itr != feature.end(); ++itr){
				float angle = *itr;
				output_dynamic_file << angle << " ";
			}
			output_dynamic_file << endl;
		//	output_feature_file.close();
		}
		output_dynamic_file.close();
	}
	return 0;
}

//身長の算出
float calc_tall(vector<vector<Point3f>> connected_positions){
	//足の中点求める
	int frameSize;
	vector<Point3f> p1 = connected_positions[JointType_FootLeft];
	vector<Point3f> p2 = connected_positions[JointType_FootRight];
	vector<Point3f> p3 = connected_positions[JointType_Head];
	int p1Size = p1.size();
	int p2Size = p2.size();
	int p3Size = p3.size();
	int min = 10000;
	if (p1Size == p2Size && p2Size == p3Size){
		frameSize = p1Size;
	}
	else {
		min = p1Size;
		frameSize = p1Size;
		if (p2Size < min){
			min = p2Size;
			frameSize = p2Size;
		}
		if (p3Size < min){
			min = p3Size;
			frameSize = p3Size;
		}
	}
	float max_tall = 4000.0;
	float min_tall = 100.0;
	float range_tall = max_tall - min_tall;
	float tall = 0.0;
	for (int i = 0; i < frameSize; i++){
		Point3f foot1 = p1[i];
		Point3f foot2 = p2[i];
		Point3f head = p3[i];
		Point3f cfoot = { (foot1.x + foot2.x) / 2, (foot1.y + foot2.y) / 2, (foot1.z + foot2.z) / 2 };
		tall += evaluate_dist(cfoot, head);
	}
	tall /= frameSize;
	return (tall - min_tall) / range_tall;
}

float calc_shoulderLen(vector<vector<Point3f>> v, vector<int> joints){
	float length = calc_length(v, joints);
	float max_length = 2000.0;
	float min_length = 50.0;
	float range_length = max_length - min_length;
	return (length - min_length) / range_length;
}

float calc_armLeftLen(vector<vector<Point3f>> v, vector<int> joints){
	float length = calc_length(v, joints);
	float max_length = 1600.0;
	float min_length = 25.0;
	float range_length = max_length - min_length;
	return (length - min_length) / range_length;
}

float calc_armRightLen(vector<vector<Point3f>> v, vector<int> joints){
	float length = calc_length(v, joints);
	float max_length = 1600.0;
	float min_length = 25.0;
	float range_length = max_length - min_length;
	return (length - min_length) / range_length;
}

float calc_bodyLen(vector<vector<Point3f>> v, vector<int> joints){
	float length = calc_length(v, joints);
	float max_length = 1600.0;
	float min_length = 100.0;
	float range_length = max_length - min_length;
	return (length - min_length) / range_length;
}


//身長以外の距離算出
float calc_length(vector<vector<Point3f>> v, vector<int> joints){
	int frameSize;
	vector<Point3f> p3 = v[joints[0]];
	vector<Point3f> p2 = v[joints[1]];
	vector<Point3f> p1 = v[joints[2]];
	int p1Size = p1.size();
	int p2Size = p2.size();
	int p3Size = p3.size();
	int min = 10000;
	if (p1Size == p2Size && p2Size == p3Size){
		frameSize = p1Size;
	}
	else {
		min = p1Size;
		frameSize = p1Size;
		if (p2Size < min){
			min = p2Size;
			frameSize = p2Size;
		}
		if (p3Size < min){
			min = p3Size;
			frameSize = p3Size;
		}
	}
	float dist = 0.0;
	for (int i = 0; i < frameSize; i++){
		Point3f left = p1[i];
		Point3f right = p2[i];
		Point3f center = p3[i];
		dist += evaluate_dist(left, center) + evaluate_dist(right, center);
	}
	dist /= frameSize;
	return dist;
}

//3点を与えられたときに角度を求める
//c:角度の基準点、a,b:それ以外
float evaluate_angle(Point3f c, Point3f a, Point3f b, int freq ,int joint)
{
	int ax_cx = c.x - a.x;
	int ay_cy = c.y - a.y;
	int az_cz = c.z - a.z;
	int bx_cx = c.x - b.x;
	int by_cy = c.y - b.y;
	int bz_cz = c.z - b.z;
	float cos = ((ax_cx*bx_cx) + (ay_cy*by_cy) + (az_cz*bz_cz)) / ((sqrt((ax_cx*ax_cx) + (ay_cy*ay_cy) + (az_cz*az_cz))*sqrt((bx_cx*bx_cx) + (by_cy*by_cy) + (bz_cz*bz_cz))));
	cout << freq << " " << joint << " " << cos << endl;
	float angle = acosf(cos);
	if (cos > -1.0 && cos < 0.0){
		angle = PI - angle; 
	}
	return angle;
}

//二つのベクトルのスタートが離れている場合（腰の角度算出に使用）
//膝がAorC,骨盤がBorD
float evaluate_seperated_angle(Point3f pA, Point3f pB, Point3f pC, Point3f pD)
{
	int ax_bx = pA.x - pB.x;
	int ay_by = pA.y - pB.y;
	int az_bz = pA.z - pB.z;
	int cx_dx = pC.x - pD.x;
	int cy_dy = pC.y - pD.y;
	int cz_dz = pC.z - pD.z;
	float cos = ((ax_bx*cx_dx) + (ay_by*cy_dy) + (az_bz*cz_dz)) / ((sqrt((ax_bx*ax_bx) + (ay_by*ay_by) + (az_bz*az_bz))*sqrt((cx_dx*cx_dx) + (cy_dy*cy_dy) + (cz_dz*cz_dz))));
	float angle = acosf(cos);
	if (cos > -1.0 && cos < 0.0){
		angle = PI - angle;
	}
	return angle;
}

//ユークリッド距離算出
float evaluate_dist(Point3f a, Point3f b){
	float d = sqrtf((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z));
	return d;
}