#Used Dr.Smays Stem File
#Used ChatGPT for debug and code check during replacement of missing code

#region imports
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg

#these imports are necessary for drawing a matplot lib graph on my GUI
#no simple widget for this exists in QT Designer, so I have to add the widget in code.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height/2
        self.left = self.x - self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)  # Red color pen
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.top = -self.height/2
        self.left = -self.width/2
        self.rect=qtc.QRectF( self.left, self.top, self.width, self.height)
        painter.drawRect(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()
        # brPen=qtg.QPen()
        # brPen.setWidth(0)
        # painter.setPen(brPen)
        # painter.setBrush(qtc.Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

class Wheel(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None, name='Wheel', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = wheelBrush
        self.radius = radius
        self.rect = qtc.QRectF(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)
        self.massBlock = MassBlock(CenterX, CenterY, width=2*radius*0.85, height=radius/3, pen=pen, brush=massBrush, name="Wheel Mass", mass=mass)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect
    def addToScene(self, scene):
        scene.addItem(self)
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)  # Red color pen
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.rect=qtc.QRectF(-self.radius, -self.radius, self.radius*2, self.radius*2)
        painter.drawEllipse(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()
        # brPen=qtg.QPen()
        # brPen.setWidth(0)
        # painter.setPen(brPen)
        # painter.setBrush(qtc.Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

#endregion

#region custom graphics items for schematic
class SpringItem(qtw.QGraphicsItem):
    def __init__(self, x1, y1, x2, y2, coils=5, amplitude=5, pen=None, parent=None):
        super().__init__(parent)
        self.start = qtc.QPointF(x1, y1)
        self.end   = qtc.QPointF(x2, y2)
        self.coils = coils
        self.amp   = amplitude
        self.pen   = pen or qtg.QPen(qtc.Qt.black)
        # build a coiled path
        path = qtg.QPainterPath(self.start)
        dx = (self.end.x() - self.start.x()) / (2*self.coils)
        dy = (self.end.y() - self.start.y()) / (2*self.coils)
        for i in range(1, 2*self.coils + 1):
            nx = self.start.x() + dx*i
            ny = self.start.y() + dy*i
            if i % 2:
                # perpendicular offset
                normal = qtc.QLineF(self.start, self.end).normalVector().unitVector().p2()
                perp   = qtc.QPointF(normal.x()*self.amp, normal.y()*self.amp)
                path.lineTo(nx + perp.x(), ny + perp.y())
            else:
                path.lineTo(nx, ny)
        self.path = path

    def boundingRect(self):
        return self.path.boundingRect()

    def paint(self, painter, option, widget=None):
        painter.setPen(self.pen)
        painter.drawPath(self.path)

class DashpotItem(qtw.QGraphicsItem):
    def __init__(self, x1, y1, x2, y2, width=12, pen=None, brush=None, parent=None):
        super().__init__(parent)
        # endpoints
        self.start = qtc.QPointF(x1, y1)
        self.end   = qtc.QPointF(x2, y2)
        # overall width of the damper body
        self.width = width
        self.pen   = pen   or qtg.QPen(qtc.Qt.black)
        self.brush = brush or qtg.QBrush(qtc.Qt.lightGray)

        # calculate the outer body rectangle
        top    = min(self.start.y(), self.end.y())
        height = abs(self.end.y() - self.start.y())
        left   = self.start.x() - self.width/2
        self.body_rect = qtc.QRectF(left, top, self.width, height)

        # calculate the inner piston housing (narrower rectangle)
        rod_w = self.width * 0.4
        rod_left = left + (self.width - rod_w)/2
        self.rod_rect = qtc.QRectF(rod_left, top, rod_w, height)

        # rod extension beyond the bottom
        self.extension = 10  # pixels

    def boundingRect(self):
        # include body, rod, and extension
        br = self.body_rect.united(self.rod_rect)
        br = br.united(
            qtc.QRectF(
                self.rod_rect.center().x(),
                self.rod_rect.bottom(),
                0,
                self.extension
            )
        )
        return br.adjusted(-2, -2, 2, 2)

    def paint(self, painter, option, widget=None):
        # draw outer damper body
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawRect(self.body_rect)

        # draw inner piston housing
        inner_brush = qtg.QBrush(qtc.Qt.darkGray)
        painter.setBrush(inner_brush)
        painter.drawRect(self.rod_rect)

        # draw the piston rod extension
        rod_x = self.rod_rect.center().x()
        rod_pen = qtg.QPen(self.pen.color(), 2)
        painter.setPen(rod_pen)
        painter.drawLine(
            qtc.QPointF(rod_x, self.rod_rect.bottom()),
            qtc.QPointF(rod_x, self.rod_rect.bottom() + self.extension)
        )

#endregion

#region MVC for quarter car model
class CarModel():
    """
    I re-wrote the quarter car model as an object oriented program
    and used the MVC pattern.  This is the quarter car model.  It just
    stores information about the car and results of the ode calculation.
    """
    def __init__(self):
        """
        self.results to hold results of odeint solution
        self.t time vector for odeint and for plotting
        self.tramp is time required to climb the ramp
        self.angrad is the ramp angle in radians
        self.ymag is the ramp height in m
        """
        self.results = []
        self.tmax = 3.0  # limit of timespan for simulation in seconds
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0  # time to traverse the ramp in seconds
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)  # ramp height in meters.  default is 0.1515 m
        self.yangdeg = 45.0  # ramp angle in degrees.  default is 45
        self.results = None

        #set default values for the properties of the quarter car model
        self.m1 = 450.0    # mass of car body in kg
        self.m2 = 20.0     # mass of wheel in kg
        self.c1 = 4500.0   # damping coefficient in N*s/m
        self.k1 = 15000.0  # spring constant of suspension in N/m
        self.k2 = 90000.0  # spring constant of tire in N/m
        self.v  = 120.0    # velocity of car in kph

        self.mink1 = self.m1 * 9.81 / 0.1524  #If I jack up my car and release the load on the spring, it extends about 3 inches
        self.maxk1 = self.m1 * 9.81 / 0.0762 #What would be a good value for a soft spring vs. a stiff spring?
        total_mass = self.m1 + self.m2
        self.mink2 = total_mass * 9.81 / 0.0381  #Same question for the shock absorber.
        self.maxk2 = total_mass * 9.81 / 0.01905
        self.accel =None
        self.accelMax = 0.0
        self.accelLim = 2.0
        self.SSE = 0.0

class CarView():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        # unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        # creating a canvas to draw a figure for the car model
        self.figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_horizontal_main.addWidget(self.canvas)

        # axes for the plotting using view
        self.ax = self.figure.add_subplot()
        if self.ax is not None:
            self.ax1 = self.ax.twinx()

        self.buildScene()

    def updateView(self, model=None):
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        stTmp="k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(model.mink1, model.maxk1, model.mink2, model.maxk2)
        stTmp+="SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        self.doPlot(model)

    def buildScene(self):
        # create the scene
        self.scene = qtw.QGraphicsScene()
        self.scene.setSceneRect(-200, -200, 400, 400)
        self.gv_Schematic.setScene(self.scene)

        # pens & brushes
        self.setupPensAndBrushes()

        # add wheel and body
        # note: Wheel constructor (CenterX, CenterY, radius)
        self.Wheel = Wheel(0, 50, 50, pen=self.penWheel, wheelBrush=self.brushWheel, massBrush=self.brushMass)
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penWheel, brush=self.brushMass, name="Car Body",
                                 mass=150)
        self.Wheel.addToScene(self.scene)
        self.scene.addItem(self.CarBody)

        # figure‐out key Y positions
        wheel_top_y = self.Wheel.y - self.Wheel.radius  # 50 - 50 =   0
        wheel_bottom_y = self.Wheel.y + self.Wheel.radius  # 50 + 50 = 100
        body_bottom_y = self.CarBody.y + self.CarBody.height / 2  # -70 + 15 = -55
        road_y = wheel_bottom_y + 20  # 120

        # 1) suspension spring k1 between body bottom and wheel top
        spring1 = SpringItem(
            0, body_bottom_y,
            0, wheel_top_y,
            coils=8,
            amplitude=7,
            pen=self.penWheel
        )
        self.scene.addItem(spring1)

        # 2) dashpot c1 (simple rectangle) just to the right of the spring
        dash1 = DashpotItem(
            12, body_bottom_y,
            12, wheel_top_y,
            width=10,
            pen=self.penWheel,
            brush=self.brushMass
        )
        self.scene.addItem(dash1)

        # 3) tire spring k2 between wheel bottom and road
        spring2 = SpringItem(
            0, wheel_bottom_y,
            0, road_y,
            coils=6,
            amplitude=5,
            pen=self.penWheel
        )
        self.scene.addItem(spring2)

        # 4) thick black road line
        roadPen = qtg.QPen(qtc.Qt.black)
        roadPen.setWidth(3)
        roadLine = qtw.QGraphicsLineItem(-200, road_y, 200, road_y)
        roadLine.setPen(roadPen)
        self.scene.addItem(roadLine)

    def setupPensAndBrushes(self):
        self.penWheel = qtg.QPen(qtg.QColor("orange"))
        self.penWheel.setWidth(1)
        self.brushWheel = qtg.QBrush(qtg.QColor.fromHsv(35,255,255, 64))
        self.brushMass = qtg.QBrush(qtg.QColor(200,200,200, 128))

    def doPlot(self, model=None):
        if model.results is None:
            return
        ax = self.ax
        ax1=self.ax1
        # plot result of odeint solver
        QTPlotting = True  # assumes we are plotting onto a QT GUI form
        if ax == None:
            ax = plt.subplot()
            ax1=ax.twinx()
            QTPlotting = False  # actually, we are just using CLI and showing the plot
        ax.clear()
        ax1.clear()
        t=model.t
        ycar = model.results[:,0]
        ywheel=model.results[:,2]
        accel=model.accel

        if self.chk_LogX.isChecked():
            ax.set_xlim(0.001,model.tmax)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0.0, model.tmax)
            ax.set_xscale('linear')

        if self.chk_LogY.isChecked():
            ax.set_ylim(0.0001,max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('log')
        else:
            ax.set_ylim(0.0, max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('linear')

        ax.plot(t, ycar, 'b-', label='Body Position')
        ax.plot(t, ywheel, 'r-', label='Wheel Position')
        if self.chk_ShowAccel.isChecked():
            ax1.plot(t, accel, 'g-', label='Body Accel')
            ax1.axhline(y=accel.max(), color='orange')  # horizontal line at accel.max()
            ax1.set_yscale('log' if self.chk_LogAccel.isChecked() else 'linear')

        # add axis labels
        ax.set_ylabel("Vertical Position (m)", fontsize='large' if QTPlotting else 'medium')
        ax.set_xlabel("time (s)", fontsize='large' if QTPlotting else 'medium')
        ax1.set_ylabel("Y'' (g)", fontsize = 'large' if QTPlotting else 'medium')
        ax.legend()

        ax.axvline(x=model.tramp)  # vertical line at tramp
        ax.axhline(y=model.ymag)  # horizontal line at ymag
        # modify the tick marks
        ax.tick_params(axis='both', which='both', direction='in', top=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        ax1.tick_params(axis='both', which='both', direction='in', right=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        # show the plot
        if QTPlotting == False:
            plt.show()
        else:
            self.canvas.draw()

class CarController():
    def __init__(self, args):
        """
        This is the controller I am using for the quarter car model.
        """
        self.input_widgets, self.display_widgets = args
        #unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        self.model = CarModel()
        self.view = CarView(args)

        self.chk_IncludeAccel=qtw.QCheckBox()

    def ode_system(self, X, t):
        # define the forcing function equation for the linear ramp
        # It takes self.tramp time to climb the ramp, so y position is
        # a linear function of time.
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        x1    = X[0]  # car position in vertical direction
        x1dot = X[1]  # car velocity in vertical direction
        x2    = X[2]  # wheel position in vertical direction
        x2dot = X[3]  # wheel velocity in vertical direction

        # write the non-trivial equations in vertical direction
        # suspension force (spring + damper)
        Fs = self.model.c1*(x1dot - x2dot) + self.model.k1*(x1 - x2)
        # tire spring force
        Ft = self.model.k2*(x2 - y)

        x1ddot = -Fs / self.model.m1
        x2ddot =  (Fs - Ft) / self.model.m2

        # return the derivatives of the input state vector
        return [x1dot, x1ddot, x2dot, x2ddot]


    def calculate(self, doCalc=True):
        """
        I will first set the basic properties of the car model and then calculate the result
        in another function doCalc.
        """
        #Step 1.  Read from the widgets
        self.model.m1  = float(self.le_m1.text())
        self.model.m2  = float(self.le_m2.text())
        self.model.c1  = float(self.le_c1.text())
        self.model.k1  = float(self.le_k1.text())
        self.model.k2  = float(self.le_k2.text())
        self.model.v   = float(self.le_v.text())

        #recalculate min and max k values
        self.mink1     = self.model.m1 * 9.81 / 0.1524
        self.maxk1     = self.model.m1 * 9.81 / 0.0762
        self.mink2     = (self.model.m1 + self.model.m2) * 9.81 / 0.0381
        self.maxk2     = (self.model.m1 + self.model.m2) * 9.81 / 0.01905

        ymag=6.0/(12.0*3.3)   #This is the height of the ramp in m
        if ymag is not None:
            self.model.ymag = ymag
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())
        if(doCalc):
            self.doCalc()
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def setWidgets(self, w):
        self.view.setWidgets(w)
        self.chk_IncludeAccel=self.view.chk_IncludeAccel

    def doCalc(self, doPlot=True, doAccel=True):
        """
        This solves the differential equations for the quarter car model.
        :param doPlot:
        :param doAccel:
        :return:
        """
        v = 1000 * self.model.v / 3600  # convert speed to m/s from kph
        self.model.angrad = self.model.yangdeg * math.pi / 180.0  # convert angle to radians
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)  # calculate time to traverse ramp

        self.model.t=np.linspace(0,self.model.tmax,2000)
        ic = [0, 0, 0, 0]
        # run odeint solver
        self.model.results = odeint(self.ode_system, ic, self.model.t)
        if doAccel:
            self.calcAccel()
        if doPlot:
            self.doPlot()

    def calcAccel(self):
        """
        Calculate the acceleration in the vertical direction using the forward difference formula.
        """
        N=len(self.model.t)
        self.model.accel=np.zeros(shape=N)
        vel=self.model.results[:,1]
        for i in range(N):
            if i==N-1:
                h = self.model.t[i] - self.model.t[i-1]
                self.model.accel[i]=(vel[i]-vel[i-1])/(9.81*h)  # backward difference of velocity
            else:
                h = self.model.t[i + 1] - self.model.t[i]
                self.model.accel[i] = (vel[i + 1] - vel[i]) / (9.81 * h)  # forward difference of velocity
            # else:
            #     self.model.accel[i]=(vel[i+1]-vel[i-1])/(9.81*2.0*h)  # central difference of velocity
        self.model.accelMax=self.model.accel.max()
        return True

    def OptimizeSuspension(self):
        """
        Step 1:  set parameters based on GUI inputs by calling self.set(doCalc=False)
        Step 2:  make an initial guess for k1, c1, k2
        Step 3:  optimize the suspension
        :return:
        """
        #Step 1:

        self.calculate(doCalc=False)
        #Step 2:

        x0= np.array([self.model.k1, self.model.c1, self.model.k2])
        #Step 3:

        answer= minimize(self.SSE, x0, method='Nelder-Mead')
        # unpack the optimized parameters
        self.model.k1, self.model.c1, self.model.k2 = answer.x
        # update SSE (no penalties) and refresh the view
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def SSE(self, vals, optimizing=True):
        """
        Calculates the sum of square errors between the contour of the road and the car body.
        :param vals:
        :param optimizing:
        :return:
        """
        k1, c1, k2=vals  #unpack the new values for k1, c1, k2
        self.model.k1=k1
        self.model.c1=c1
        self.model.k2=k2
        self.doCalc(doPlot=False)  #solve the odesystem with the new values of k1, c1, k2
        SSE=0
        for i in range(len(self.model.results[:,0])):
            t=self.model.t[i]
            y=self.model.results[:,0][i]
            if t < self.model.tramp:
                ytarget = self.model.ymag * (t / self.model.tramp)
            else:
                ytarget = self.model.ymag
            SSE+=(y-ytarget)**2

        #some penalty functions if the constants are too small
        if optimizing:
            if k1<self.model.mink1 or k1>self.model.maxk1:
                SSE+=100
            if c1<10:
                SSE+=100
            if k2<self.model.mink2 or k2>self.model.maxk2:
                SSE+=100

            # I'm overlaying a gradient in the acceleration limit that scales with distance from a target squared.
            if self.model.accelMax>self.model.accelLim and self.chk_IncludeAccel.isChecked():
                # need to soften suspension
                SSE+=(self.model.accelMax-self.model.accelLim)**2
        self.model.SSE=SSE
        return SSE

    def doPlot(self):
        self.view.doPlot(self.model)
#endregion
#endregion

def main():
    QCM = CarController()
    QCM.doCalc()

if __name__ == '__main__':
    main()
