# Help test FutsalCut / 테스트에 참여하기

FutsalCut has been evaluated on one recording so far. Results from other cameras, phones and computers are the most useful thing you can send.
지금까지 영상 1개로만 평가했습니다. 다른 카메라·휴대폰·컴퓨터에서의 결과가 가장 큰 도움이 됩니다.

**You do not need to share your video / 영상은 보내지 않아도 됩니다.** The app's diagnostics are enough.

## What to do / 방법
1. Open the app in Chrome or Edge ([English](https://kachy2.github.io/futsalcut/) · [한국어](https://kachy2.github.io/futsalcut/ko/)). 앱을 열고 영상을 선택합니다.
2. Draw the zone, press **Analyze**, note the speed shown while it runs (e.g. "12.3x realtime").
   구역을 그리고 **분석 시작**을 누른 뒤, 진행 중 표시되는 속도(배속)를 메모합니다.
   The first visit, and every forced refresh (Ctrl+F5), reloads the page once more by itself (it switches on multi-threading); that is expected. If you drew the zone wrongly, use **Redraw the zone** on the result page: the video is not read again.
   처음 방문하거나 강제 새로고침(Ctrl+F5)을 하면 페이지가 한 번 더 저절로 새로고침됩니다(멀티스레드를 켜는 과정이라 정상입니다). 구역을 잘못 그렸다면 결과 화면의 **구역 다시 그리기**를 누르세요. 영상을 처음부터 다시 읽지 않습니다.
3. Check the scenes and save the MP4. Play it: is the audio in sync, do the scenes start where you expect?
   장면을 확인하고 MP4로 저장한 뒤 재생해 봅니다(음성 싱크, 장면 시작 위치).
4. Open **Diagnostics**, press **Copy**, and paste it into a [new issue](https://github.com/kachy2/futsalcut/issues/new/choose).
   **진단 정보**를 펼쳐 **복사**하고 [새 이슈](https://github.com/kachy2/futsalcut/issues/new/choose)에 붙여 넣습니다.

The person detector picks its hardware by itself (**Auto**): the graphics chip (WebGPU) when the browser offers a real one and it gives the same results as the CPU without being slower, otherwise the CPU (WASM, several threads). Under the zone editor you can force either one. The diagnostics show what was used (`detectorBackend`), the check (`detectorGpuCheck`: result difference and time per frame of the graphics chip and the CPU) and the speed (`detectorMsPerCall`); **detector test (auto)** repeats the check, and the WASM / WebGPU buttons test one backend.
사람 검출은 하드웨어를 스스로 고릅니다(**자동**). 브라우저가 진짜 그래픽 칩을 제공하고 CPU와 같은 결과를 내며 더 느리지 않으면 그래픽 칩(WebGPU)을, 아니면 CPU(WASM, 멀티스레드)를 씁니다. 구역 편집 화면에서 어느 쪽이든 강제로 고를 수 있습니다. 진단 정보에는 실제로 쓴 방식(`detectorBackend`), 확인 결과(`detectorGpuCheck`: 결과 차이와 그래픽 칩·CPU의 프레임당 시간), 속도(`detectorMsPerCall`)가 나옵니다. **detector test (auto)** 버튼은 같은 확인을 다시 하고, WASM / WebGPU 버튼은 한쪽만 시험합니다.

## Please also tell us / 함께 알려 주세요
- Camera or phone model, codec (H.264 or HEVC), resolution, number of files, length. 카메라·휴대폰 모델, 코덱, 해상도, 파일 수, 길이.
- In Chrome, open `chrome://gpu` and copy the **Video Decode** line (hardware accelerated or software).
  Chrome에서 `chrome://gpu`를 열어 **Video Decode** 줄(하드웨어/소프트웨어)을 알려 주세요.
- Did the kept scenes match what you wanted? What did it miss or wrongly keep?
  남긴 장면이 원하던 것과 맞았나요? 놓치거나 잘못 남긴 장면은?

## Phones / 휴대폰
Mobile is experimental. Chrome for Android 132 or newer reportedly enables the save dialog, but this has not been verified; tell us whether saving worked.
모바일은 실험적입니다. Android Chrome 132 이상에서 저장 대화상자가 켜졌다고 하지만 확인되지 않았으니, 저장이 되는지 알려 주세요.

## Privacy / 개인정보
The diagnostics contain your browser, CPU cores, graphics adapter name and timing numbers, plus the **file names**, sizes, codec and resolution of the selected files.
They contain no video frames. File names can reveal team names or dates: rename them or edit the `name` lines before posting.
진단 정보에는 브라우저, CPU 코어 수, 그래픽 장치 이름, 속도 수치, 선택한 파일의 **이름**·크기·코덱·해상도가 들어 있고 영상 화면은 없습니다. 파일 이름에 팀명이나 날짜가 있으면 올리기 전에 `name` 줄을 고쳐 주세요.
