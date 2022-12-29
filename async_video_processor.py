import cv2
import asyncio
import sys
import math
from PIL import Image
from timeit import default_timer
import traceback

class AsyncVideoProcessor:



    def __init__(self, inputMovie, processFunc, resultFunc, writeResultsFunc, firstFrameToProcess=0, processFps=8):
        self.inputMovie = inputMovie
        video = cv2.VideoCapture(self.inputMovie)
        if not video.isOpened():
            raise ValueError("cannot open " + inputMovie)

        self.processFunc = processFunc
        self.resultFunc = resultFunc
        self.writeResultsFunc = writeResultsFunc

        video.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        self.totalFrames = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

        self.video = video

        fps = video.get(cv2.CAP_PROP_FPS)
        duration = float(self.totalFrames) / float(fps)
        print('duration %f s, fps %f, %i frames' % (duration, fps, self.totalFrames))
        self.frameIncrement = math.ceil(fps/processFps)
        print('frame increment %i -> processing at %f fps' % (self.frameIncrement, fps/float(self.frameIncrement)))

        if firstFrameToProcess < 0:
            # we have been passed the last already-existing frame
            self.nextFrameToProcess = -firstFrameToProcess + self.frameIncrement
        else:
            self.nextFrameToProcess = firstFrameToProcess

    def getNextImage(self, video, nextFrameToProcess):
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        if currentFrame > nextFrameToProcess or abs(nextFrameToProcess - currentFrame) > 30:
            # seek
            video.set(cv2.CAP_PROP_POS_FRAMES, nextFrameToProcess)
            currentFrame = nextFrameToProcess
        else:
            # advance frames one by one
            while currentFrame < nextFrameToProcess:
                ret, frame = video.read()
                if not ret:
                    break
                currentFrame += 1
        ret, frame = video.read()
        if not ret:
            return None, None

        frame_bgr = frame[...,::-1]

        return Image.fromarray(frame_bgr)


    async def produceFrames(self, queue):
        while True:
            if self.nextFrameToProcess >= self.totalFrames:
                break

            image = await self.loop.run_in_executor(None, lambda: self.getNextImage(self.video, self.nextFrameToProcess))
            #print('produced ' + frameLabel, end='\r')
            if image == None:
                break
            await queue.put([self.nextFrameToProcess, image])

            self.nextFrameToProcess += self.frameIncrement

        await queue.put(None)

    async def consumeFrames(self, queue):
        appendedCount = 0
        unsavedCount = 0
        startTime = default_timer()
        while True:
            print(" g ", end='\r')
            item = await queue.get()
            if item is None:
                print("finished                            ") # spaces to clear %ge from progress
                self.writeResultsFunc(self.video, partial=False)
                break

            frameIndex = item[0]
            image = item[1]
            print(" p frame %i (%f%%)           " %(frameIndex, float(100*frameIndex)/float(self.totalFrames)), end='\r')
            #print('consuming ' + frameLabel)
            detections = self.processFunc(image)
            print("   ", end='\r')
            self.resultFunc(frameIndex, detections)


            unsavedCount += 1
            appendedCount += 1
            if unsavedCount >= 500:
                print("saving intermediate results")
                self.writeResultsFunc(self.video, partial=True)
                unsavedCount = 0
                currentTime = default_timer()
                totalTime = (currentTime-startTime)
                fps = float(appendedCount)/totalTime
                framesRemaining = (self.totalFrames-frameIndex)/self.frameIncrement
                secondsRemaining = framesRemaining/fps
                print("processed %i frames in %f seconds (%f fps), %im %is remaining" % (appendedCount, totalTime, fps, int(secondsRemaining/60), int(secondsRemaining%60)))

            queue.task_done()

    async def run(self):
        try:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.get_event_loop()
            queue = asyncio.Queue(maxsize=20)
            producer_coro = self.produceFrames(queue)
            consumer_coro = self.consumeFrames(queue)
            await asyncio.gather(producer_coro, consumer_coro)
        except Exception as e:
            print("Unexpected error:", str(e))
            print(traceback.format_exc())
        finally:
            #self.loop.close()
            pass
