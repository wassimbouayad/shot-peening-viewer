import pygame
from pygame.locals import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras.models import load_model
from tensorflow import reduce_min, reduce_max
import matplotlib.backends.backend_agg as agg
import pylab
from modules.integration import calculate_integration_weights, get_z, adapt_integration_constants

FPS = 25
WINDOW_MULTIPLIER = 5
WINDOW_SIZE = 90
MAP_SIZE = WINDOW_SIZE * WINDOW_MULTIPLIER
WINDOW_WIDTH = 2*WINDOW_SIZE * WINDOW_MULTIPLIER
WINDOW_HEIGHT = MAP_SIZE+120
NB_CELLS = 32
CELL_SIZE = MAP_SIZE // NB_CELLS

WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (133, 0, 173)
YELLOW = (224, 224, 224)
BLACK = (0, 0, 0)


def drawGrid():
    mapSurface.fill(WHITE)
    for x in range(0, MAP_SIZE, CELL_SIZE):
        pygame.draw.line(mapSurface, GRAY, (x, 0), (x, CELL_SIZE * NB_CELLS))
    for y in range(0, MAP_SIZE, CELL_SIZE):
        pygame.draw.line(mapSurface, GRAY, (0, y), (CELL_SIZE * NB_CELLS, y))
    for i in range(pattern.shape[1]):
        for j in range(pattern.shape[2]):
            if pattern[0, i, j].sum()>=1:
                fillCell((i, j), pattern[0, i, j])


def drawLoadButtons():
    global lengthEditX, lengthEditY, lengthEditW, lengthEditH
    global thicknessEditX, thicknessEditY, thicknessEditW, thicknessEditH
    global treatmentButton1X, treatmentButton1Y, treatmentButton1W, treatmentButton1H
    global treatmentButton2X, treatmentButton2Y, treatmentButton2W, treatmentButton2H
    global treatmentButton3X, treatmentButton3Y, treatmentButton3W, treatmentButton3H
    global clearButtonX, clearButtonY, clearButtonW, clearButtonH
    global length, thickness, treatment, load

    if length is not None and thickness is not None:
        if treatment == 1:
            alpha, beta = 6.3e-4, .63e-4
        elif treatment == 2:
            alpha, beta = 8.0e-4, -3.4e-4
        elif treatment == 3:
            alpha, beta = 8.1e-4, -1.8e-4
        load = 12*(1+.33)*(alpha/2*thickness - beta) * length**2 / (thickness**4)
        load = int(load)

    loadSurface.fill(WHITE)
    height=20
    margin=5
    font = pygame.font.SysFont('arialttf', height)
    img_length = font.render('Length:', True, BLACK)
    img_thickness = font.render('Thickness:', True, BLACK)
    img_treatment = font.render('Treatment:', True, BLACK)
    img_load = font.render('Dimensionless load: {}'.format(load), True, BLACK if 10<=load<=80 else RED)
    loadSurface.blit(img_length, (0, 0))
    loadSurface.blit(img_thickness, (0, height+margin))
    loadSurface.blit(img_treatment, (0, 2*(height+margin)))
    loadSurface.blit(img_load, (0, 3*(height+margin)))
    width_texts = max(img_length.get_width(), img_thickness.get_width(), img_treatment.get_width())

    img_length = font.render('{}'.format(length if length is not None else ''), True, BLACK)
    img_thickness = font.render('{}'.format(thickness if thickness is not None else ''), True, BLACK)
    img_treatment1 = font.render('1', True, BLACK)
    img_treatment2 = font.render('2', True, BLACK)
    img_treatment3 = font.render('8', True, BLACK)
    width_numbers = max(img_length.get_width(), img_thickness.get_width())

    lengthEditX, lengthEditY, lengthEditW, lengthEditH = width_texts + margin, 0, width_numbers, height
    thicknessEditX, thicknessEditY, thicknessEditW, thicknessEditH = width_texts + margin, height+margin, width_numbers, height
    treatmentButton1X, treatmentButton1Y, treatmentButton1W, treatmentButton1H = width_texts + margin, 2*(height+margin), height, height
    treatmentButton2X, treatmentButton2Y, treatmentButton2W, treatmentButton2H = width_texts + margin + treatmentButton1W + margin, 2*(height+margin), height, height
    treatmentButton3X, treatmentButton3Y, treatmentButton3W, treatmentButton3H = width_texts + margin + treatmentButton1W + margin + treatmentButton2W + margin, 2*(height+margin), height, height

    if lengthSelected:
        pygame.draw.rect(loadSurface, YELLOW, (lengthEditX, lengthEditY, lengthEditW, lengthEditH))
    if thicknessSelected:
        pygame.draw.rect(loadSurface, YELLOW, (thicknessEditX, thicknessEditY, thicknessEditW, thicknessEditH))
    if treatment == 1:
        pygame.draw.rect(loadSurface, YELLOW, (treatmentButton1X, treatmentButton1Y, treatmentButton1W, treatmentButton1H))
    elif treatment == 2:
        pygame.draw.rect(loadSurface, YELLOW, (treatmentButton2X, treatmentButton2Y, treatmentButton2W, treatmentButton2H))
    elif treatment == 3:
        pygame.draw.rect(loadSurface, YELLOW, (treatmentButton3X, treatmentButton3Y, treatmentButton3W, treatmentButton3H))

    loadSurface.blit(img_length, (lengthEditX + (lengthEditW - img_length.get_width())/2, lengthEditY + (lengthEditH - img_length.get_height())/2))
    loadSurface.blit(img_thickness, (thicknessEditX + (thicknessEditW - img_thickness.get_width())/2, thicknessEditY + (thicknessEditH - img_thickness.get_height())/2))
    loadSurface.blit(img_treatment1, (treatmentButton1X + (treatmentButton1W - img_treatment1.get_width())/2, treatmentButton1Y + (treatmentButton1H - img_treatment1.get_height())/2))
    loadSurface.blit(img_treatment2, (treatmentButton2X + (treatmentButton2W - img_treatment2.get_width())/2, treatmentButton2Y + (treatmentButton2H - img_treatment2.get_height())/2))
    loadSurface.blit(img_treatment3, (treatmentButton3X + (treatmentButton3W - img_treatment3.get_width())/2, treatmentButton3Y + (treatmentButton3H - img_treatment3.get_height())/2))

    pygame.draw.lines(loadSurface, BLACK, closed=True, points=[(lengthEditX, lengthEditY), (lengthEditX + lengthEditW, lengthEditY), (lengthEditX + lengthEditW, lengthEditY + lengthEditH), (lengthEditX, lengthEditY + lengthEditH)])
    pygame.draw.lines(loadSurface, BLACK, closed=True, points=[(thicknessEditX, thicknessEditY), (thicknessEditX + thicknessEditW, thicknessEditY), (thicknessEditX + thicknessEditW, thicknessEditY + thicknessEditH), (thicknessEditX, thicknessEditY + thicknessEditH)])
    pygame.draw.lines(loadSurface, BLACK, closed=True, points=[(treatmentButton1X, treatmentButton1Y), (treatmentButton1X + treatmentButton1W, treatmentButton1Y), (treatmentButton1X + treatmentButton1W, treatmentButton1Y + treatmentButton1H), (treatmentButton1X, treatmentButton1Y + treatmentButton1H)])
    pygame.draw.lines(loadSurface, BLACK, closed=True, points=[(treatmentButton2X, treatmentButton2Y), (treatmentButton2X + treatmentButton2W, treatmentButton2Y), (treatmentButton2X + treatmentButton2W, treatmentButton2Y + treatmentButton2H), (treatmentButton2X, treatmentButton2Y + treatmentButton2H)])
    pygame.draw.lines(loadSurface, BLACK, closed=True, points=[(treatmentButton3X, treatmentButton3Y), (treatmentButton3X + treatmentButton3W, treatmentButton3Y), (treatmentButton3X + treatmentButton3W, treatmentButton3Y + treatmentButton3H), (treatmentButton3X, treatmentButton3Y + treatmentButton3H)])

    img_length = font.render('mm'.format(length), True, BLACK)
    img_thickness = font.render('mm'.format(thickness), True, BLACK)
    loadSurface.blit(img_length, (width_texts + width_numbers + margin + margin, 0))
    loadSurface.blit(img_thickness, (width_texts + width_numbers + margin + margin, height+margin))

    img = pygame.font.SysFont('arialttf', 30).render('Clear', True, BLACK)
    clearButtonW, clearButtonH = img.get_size()
    clearButtonX = MAP_SIZE - clearButtonW - 10
    clearButtonY = (WINDOW_HEIGHT - MAP_SIZE - clearButtonH) / 2
    pygame.draw.rect(loadSurface, YELLOW, (clearButtonX, clearButtonY, clearButtonW, clearButtonH))
    pygame.draw.lines(loadSurface, BLACK, closed=True, points=[(clearButtonX, clearButtonY), (clearButtonX + clearButtonW, clearButtonY), (clearButtonX + clearButtonW, clearButtonY + clearButtonH), (clearButtonX, clearButtonY + clearButtonH)])
    loadSurface.blit(img, (clearButtonX, clearButtonY))


def drawScaleButtons():
    scaleSurface.fill(WHITE)
    font = pygame.font.SysFont('arialttf', 30)
    img = font.render('Zoom', True, BLACK)
    scaleSurface.blit(img, ((MAP_SIZE - img.get_width()) / 2, 0))
    x, y, w, h = MAP_SIZE / 2 - 10 - 40, 40, 40, 40
    pygame.draw.rect(scaleSurface, YELLOW, (x, y, w, h))
    pygame.draw.lines(scaleSurface, BLACK, closed=True, points=[(x,y), (x,y+h), (x+w,y+h), (x+w,y)])
    img = font.render('-', True, BLACK)
    scaleSurface.blit(img, (x+w/2-img.get_width()/2, y+h/2-img.get_height()/2-3))
    x = MAP_SIZE / 2 + 10
    pygame.draw.rect(scaleSurface, YELLOW, (x, y, w, h))
    pygame.draw.lines(scaleSurface, BLACK, closed=True, points=[(x,y), (x,y+h), (x+w,y+h), (x+w,y)])
    img = font.render('+', True, BLACK)
    scaleSurface.blit(img, (x+w/2-img.get_width()/2, y+h/2-img.get_height()/2))


def getButton(mousex, mousey):
    if clearButtonX < mousex < clearButtonX + clearButtonW and MAP_SIZE + clearButtonY < mousey < MAP_SIZE + clearButtonY + clearButtonH:
        return 0
    elif lengthEditX < mousex < lengthEditX + lengthEditW and MAP_SIZE + lengthEditY < mousey < MAP_SIZE + lengthEditY + lengthEditH:
        return 1
    elif thicknessEditX < mousex < thicknessEditX + thicknessEditW and MAP_SIZE + thicknessEditY < mousey < MAP_SIZE + thicknessEditY + thicknessEditH:
        return 2
    elif treatmentButton1X < mousex < treatmentButton1X + treatmentButton1W and MAP_SIZE + treatmentButton1Y < mousey < MAP_SIZE + treatmentButton1Y + treatmentButton1H:
        return 3
    elif treatmentButton2X < mousex < treatmentButton2X + treatmentButton2W and MAP_SIZE + treatmentButton2Y < mousey < MAP_SIZE + treatmentButton2Y + treatmentButton2H:
        return 4
    elif treatmentButton3X < mousex < treatmentButton3X + treatmentButton3W and MAP_SIZE + treatmentButton3Y < mousey < MAP_SIZE + treatmentButton3Y + treatmentButton3H:
        return 5
    elif MAP_SIZE+MAP_SIZE/2-10-40 < mousex < MAP_SIZE+MAP_SIZE/2-10 and MAP_SIZE+40 < mousey < MAP_SIZE+40+40:
        return 6
    elif MAP_SIZE+MAP_SIZE/2+10 < mousex < MAP_SIZE+MAP_SIZE/2+10+40 and MAP_SIZE+40 < mousey < MAP_SIZE+40+40:
        return 7
    return -1


def buttonClicked(buttonid):
    global treatment, extent, length, thickness, lengthSelected, thicknessSelected
    global pattern_changed
    lengthSelected, thicknessSelected = False, False
    if buttonid == -1:
        return
    if buttonid == 0:
        # clear
        pattern[:,:,:,:] = 0
        drawGrid()
        pattern_changed = True
        return
    if buttonid == 1:
        length=None
        lengthSelected = True
        pattern_changed = False
        drawLoadButtons()
        return
    if buttonid == 2:
        thickness=None
        thicknessSelected = True
        pattern_changed = False
        drawLoadButtons()
        return
    if buttonid == 3:
        treatment = 1
        drawLoadButtons()
        pattern_changed = True
        return
    if buttonid == 4:
        treatment = 2
        drawLoadButtons()
        pattern_changed = True
        return
    if buttonid == 5:
        treatment = 3
        drawLoadButtons()
        pattern_changed = True
        return
    if buttonid == 6:
        # scale +
        extent *= 2
        pattern_changed = True
        return
    if buttonid == 7:
        # scale -
        extent /= 2
        pattern_changed = True
        return


def getCellIndexes(mousex, mousey):
    i = mousex * NB_CELLS // MAP_SIZE
    j = mousey * NB_CELLS // MAP_SIZE
    if i>=0 and i<NB_CELLS and j>=0 and j<NB_CELLS:
        return i, j
    else:
        return None


def highlightCell(mouseCell, previousMouseCell=None):
    if previousMouseCell is not None:
        i, j = previousMouseCell
        boxx = i * CELL_SIZE
        boxy = j * CELL_SIZE
        if pattern[0, i, j].sum()>=1:
            fillCell((i, j), pattern[0, i, j])
        else:
            pygame.draw.rect(mapSurface, WHITE, (boxx + 1, boxy + 1, CELL_SIZE - 1, CELL_SIZE - 1), 1)
    if mouseCell is None:
        return
    i, j = mouseCell
    boxx = i * CELL_SIZE
    boxy = j * CELL_SIZE
    pygame.draw.rect(mapSurface, YELLOW, (boxx + 1, boxy + 1, CELL_SIZE - 1, CELL_SIZE - 1), 1)


def fillCell(cell, pat):
    i, j = cell
    boxx = i * CELL_SIZE
    boxy = j * CELL_SIZE
    if pat[0] == 1 and pat[1] == 0:
        color = RED
    elif pat[0] == 0 and pat[1] == 1:
        color = BLUE
    elif pat[0] == 1 and pat[1] == 1:
        color = PURPLE
    else:
        color = WHITE
    pygame.draw.rect(mapSurface, color, (boxx + 1, boxy + 1, CELL_SIZE - 1, CELL_SIZE - 1))


def cellClicked(mouseCell, mode):
    global pattern_changed
    if mouseCell is None:
        return
    i, j = mouseCell
    if mode==-1:
        pattern[0,i,j,0] = 0
        pattern[0,i,j,1] = 0
    else:
        pattern[0, i, j, mode] = 1
    fillCell((i, j), pattern[0, i, j])
    pattern_changed=True


def get_z_from_NN():
    # std = [0.00371526, 0.00147998, 0.00371526]
    # mean = [3.83101875e-06, 2.41404390e-22, 3.83101875e-06]
    std = [1.5e-2, 7.5e-3, 1.5e-2]
    mean = [0, 0, 0]

    rot_pattern = np.concatenate([pattern, np.rot90(pattern, axes=(1,2))])
    rot_pattern = np.concatenate([rot_pattern, np.rot90(rot_pattern, k=2, axes=(1,2))])
    sym_rot_pattern = np.concatenate([rot_pattern, rot_pattern[:,:,:,[1,0]]])
    out = (net(sym_rot_pattern*load/80)*std+mean).numpy()
    out[[4,5,6,7]] = -out[[4,5,6,7]]
    out[[1,5]] = np.rot90(out[[1,5]], k=3, axes=(1,2))
    out[[2,6]] = np.rot90(out[[2,6]], k=2, axes=(1,2))
    out[[3,7]] = np.rot90(out[[3,7]], k=1, axes=(1,2))
    out[[1,3,5,7]] = out[[1,3,5,7]][:,:,:,[2,1,0]]
    out[[1,3,5,7],:,:,1] = -out[[1,3,5,7],:,:,1]
    # import matplotlib.pyplot as plt
    # for i in range(8):
    #     plt.figure()
    #     plt.imshow(out[i,:,:,0])
    # plt.show()
    # print(out)
    # print(out.std(0).mean(), out[0].std())
    out = np.expand_dims(out.mean(0), 0)

    return adapt_integration_constants(get_z(out, integration_weights)[0], np.zeros((32,32,1)))


def drawResult(pattern_changed=True):
    global ax, fig, length
    if pattern_changed:
        # close the previous figure
        plt.close('all')
        # get net output
        res = get_z_from_NN()[:,:,0] * length / 32

    #    create the figure
        fig = pylab.figure(figsize=(4,4), dpi=100)
        ax = fig.gca(projection='3d')
        X = np.linspace(0, length, 32)
        Y = np.linspace(0, length, 32)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, res, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        med = (res.min() + res.max()) / 2
        min, max = med - extent/2, med + extent/2
        ax.set_zlim(min, max)
        fig.colorbar(surf, shrink=.5, aspect=5)#, ticks=np.arange(-1.1, .1, .1))
    ax.view_init(30,angle)

#    plot the figure
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), 'RGB')
    resultSurface.blit(surf, (0,0))


def main(model_path='saved_models/unet_pattern_to_K'):
    global mapSurface, resultSurface, loadSurface, scaleSurface
    global net, pattern, integration_weights
    global length, thickness, treatment, load
    global lengthSelected, thicknessSelected
    global angle, extent, pattern_changed
    length = 100
    thickness = 1
    treatment = 1
    load = None
    lengthSelected, thicknessSelected = False, False
    angle=0
    count=0
    extent=1.

    pattern = np.zeros((1, NB_CELLS, NB_CELLS, 2))
    # model_path='saved_models/unet_e11_to_coord_v1'
    net = load_model(model_path)
    print('Model used:', model_path)
    integration_weights = calculate_integration_weights()

    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    displaySurface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))#, RESIZABLE)
    pygame.display.set_caption('Shot-peening')

    displaySurface.fill(WHITE)
    mapSurface = displaySurface.subsurface(pygame.Rect(0, 0, MAP_SIZE, MAP_SIZE))
    resultSurface = displaySurface.subsurface(pygame.Rect(MAP_SIZE, 0, WINDOW_WIDTH - MAP_SIZE, WINDOW_HEIGHT))
    loadSurface = displaySurface.subsurface(pygame.Rect(0, MAP_SIZE, MAP_SIZE, WINDOW_HEIGHT - MAP_SIZE))
    scaleSurface = displaySurface.subsurface(pygame.Rect(MAP_SIZE, MAP_SIZE, WINDOW_WIDTH - MAP_SIZE, WINDOW_HEIGHT - MAP_SIZE))
    drawGrid()
    drawLoadButtons()
    drawResult()
    drawScaleButtons()
    pygame.display.flip()

    mouseCell = None
    highlightedCell = None
    mouseClicking = False
    mode = 0
    quit=False
    pattern_changed=False
    while True:
        for event in pygame.event.get():
            # quit event
            if event.type == KEYDOWN:
                if (lengthSelected or thicknessSelected) and event.unicode.isdigit():
                    c = event.unicode
                    if lengthSelected:
                        if length is not None:
                            length = 10*length + int(c)
                        else:
                            length = int(c)
                            if length == 0:
                                length = None
                    elif thicknessSelected:
                        if thickness is not None:
                            thickness = 10*thickness + int(c)
                        else:
                            thickness = int(c)
                            if thickness == 0:
                                thickness = None
                    drawLoadButtons()
                elif (lengthSelected or thicknessSelected) and event.key == K_RETURN:
                    lengthSelected, thicknessSelected = False, False
                    pattern_changed = True
                    if length == None:
                        length = 100
                    if thickness == None:
                        thickness = 1
                    drawLoadButtons()
                if (event.mod == KMOD_LGUI) | (event.mod == KMOD_GUI) | (event.mod == KMOD_CTRL) | (event.mod == KMOD_LCTRL):
                    if event.key == K_w:
                        quit=True
                elif (event.key == K_LSHIFT) | (event.key == K_RSHIFT):
                    mode = 1
            elif event.type == KEYUP:
                if (event.key == K_LSHIFT) | (event.key == K_RSHIFT):
                    mode = 0
            if event.type == QUIT:
                quit=True
            if quit:
                pygame.quit()
                sys.exit()
            # events other than quit
            if event.type == MOUSEMOTION:
                mouseCell = getCellIndexes(*event.pos)
                if mouseClicking:
                    cellClicked(mouseCell, mode)
                else:
                    highlightCell(mouseCell, highlightedCell)
                    highlightedCell = mouseCell
            elif event.type == MOUSEBUTTONDOWN:
                mouseClicking = True
                mouseCell = getCellIndexes(*event.pos)
                cellClicked(mouseCell, mode)
                highlightedCell = None
            elif event.type == MOUSEBUTTONUP:
                mouseClicking = False
                mouseCell = getCellIndexes(*event.pos)
                cellClicked(mouseCell, mode)
                mouseButton = getButton(*event.pos)
                buttonClicked(mouseButton)
                drawGrid()
                drawResult(pattern_changed=pattern_changed)
                pattern_changed = False
                drawLoadButtons()
        if count % 10 == 0:
            angle += 1
            drawResult(pattern_changed=pattern_changed)
        pygame.display.update()
        FPS_CLOCK.tick(FPS)
        count += 1

if __name__=='__main__':
    a=main()
