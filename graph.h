#ifndef OLC_PGE_APPLICATION
#define OLC_PGE_APPLICATION

#include <vector>
#include <thread>
#include "olcPixelGameEngine/olcPixelGameEngine.h"

#include <stdio.h>
#include <iostream>

using namespace std;


struct Point {
	int x = 0, y = 0;

	Point (int x, int y) : x(x), y(y) {

	}
};

struct Line {
	vector<Point> points;

	void addPoint(const Point& p) {
		points.push_back(p);
	}
};











struct Graph : public olc::PixelGameEngine {

	int w, h, pixelW, pixelH;

	vector<Point> points;
	vector<Line*> lines;

	std::thread graphThread;

	Graph(int w = 800, int h = 650, int pixelW = 1, int pixelH = 1) : w(w), h(h), pixelW(pixelW), pixelH(pixelH) {

		// start the graph in another thread so the rest of the code doesn't stop
		graphThread = std::thread([=]() {
			if (Construct(w, h, pixelW, pixelH)) {
				Start();
			}
		});
	}

	bool OnUserCreate() override {
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override {
		
		Clear(olc::BLACK);

		for (size_t i = 0; i < lines.size(); ++i) {
			drawLine(lines[i]);
		}

		for (size_t i = 0; i < points.size(); ++i) {
			drawPoint(points[i], 2);
		}

		return true;
	}

	void addLine(Line* line) {
		lines.push_back(line);
	}

	void drawLine(Line* line) {
		Line l = *line;
		for (size_t i = 0; i + 1 < l.points.size(); ++i) {
			DrawLine(l.points[i].x, h / 2 - l.points[i].y, l.points[i + 1].x, h / 2 - l.points[i + 1].y);
		}
	}

	void addPoint(Point& p) {
		points.push_back(p);
	}

	void drawPoint(Point& p, int size = 1) {
		FillCircle(p.x, p.y, size);
	}


	void waitFinish() {
		graphThread.join();
	}

};







#endif