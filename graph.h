#ifndef OLC_PGE_APPLICATION
#define OLC_PGE_APPLICATION

#include <vector>
#include <thread>
#include "olcPixelGameEngine/olcPixelGameEngine.h"

#include <stdio.h>
#include <iostream>

using namespace std;


struct Point {
	double x, y;
	olc::Pixel color = olc::BLACK;

	Point (double x = 0, double y = 0) : x(x), y(y) {
		
	}
};

struct Line {
	vector<Point> points;
	olc::Pixel color;

	Line(olc::Pixel color = olc::BLACK) : color(color) {

	}

	void addPoint(const Point& p) {
		points.push_back(p);
	}
};











struct Graph : public olc::PixelGameEngine {

private:

	double minX, maxX, minY, maxY;
	std::thread graphThread;

public:

	int w, h, pixelW, pixelH;

	vector<Point> points;
	vector<Line*> lines;


	Graph(int w = 800, int h = 650, int pixelW = 1, int pixelH = 1) : w(w), h(h), pixelW(pixelW), pixelH(pixelH) {

		minX = -3;
		maxX = 7;
		minY = -3 * ((double) h / w);
		maxY = 7 * ((double) h / w);

		cout << minY << ", " << (double) w / h << "\n";

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
		
		Clear(olc::WHITE);

		drawAxis();

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
			// transform to screen space
			Point p1 = pointToScreenSpace(l.points[i]);
			Point p2 = pointToScreenSpace(l.points[i + 1]);

			DrawLine(p1.x, p1.y, p2.x, p2.y, l.color);
		}
	}

	void addPoint(Point& p) {
		points.push_back(p);
	}

	void drawPoint(Point& p, int size = 1) {
		Point screenSpace = pointToScreenSpace(p);
		FillCircle(screenSpace.x, screenSpace.y, size, olc::BLACK);
	}


	void waitFinish() {
		graphThread.join();
	}


	void drawAxis() {
		int x0 = -minX * (w / (maxX - minX));
		DrawLine(x0, 0, x0, h, olc::BLACK);

		int y0 = h + minY * (h / (maxY - minY));
		DrawLine(0, y0, w, y0, olc::BLACK);
	}


	Point pointToScreenSpace(Point p) {
		Point screenSpace;
		screenSpace.x = (p.x - minX) * (w / (maxX - minX));
		screenSpace.y = h - (p.y - minY) * (h / (maxY - minY));

		return screenSpace;

		// we need to map X coordinate from [minX, maxX] to [0, w] (and similar with Y)



		return screenSpace;
	}

};







#endif
