import { Card, Flex, Box, TextField, IconButton, Heading, Grid, Text, Slider, Avatar, Spinner, AlertDialog, Button } from '@radix-ui/themes';
import { useContext, useEffect, useRef, useState } from 'react';

import { Link } from 'react-router-dom';
import { CallbackFunction, PythonShellOptions } from 'services/pythonService';
import { AnalysisContext } from './session-analysis';
import { AllMapsBasicInfoListDto, MapOption, UploadReacingSessionInitDto, UploadReacingSessionInitReturnDto } from 'data/live-analysis/live-analysis-type';
import apiService from 'services/api.service';
import { useAuth } from 'hooks/AuthProvider';
import { ACC_STATUS, ACCMemoeryTracks } from 'data/live-analysis/live-map-data';
import { Cross2Icon } from '@radix-ui/react-icons';
import { IpcRendererEvent } from 'electron';


const LiveAnalysisSessionRecording = () => {
    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();

    let isCheckingLiveSession = false;
    const [hasValidLiveSession, setValidLiveSession] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [isRecordEnded, setIsRecorEnded] = useState(false);
    const [checkSessionScriptShellId, setCheckSessionScriptShellId] = useState(-1);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {

        //start checking a valid live session at start
        startCheckingLiveSessionInterval();

        return () => {
            // Clean up listeners when component unmounts
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    /**
     * check acc memory once and see if there is a valid session running
     */
    const CheckSessionValid = async () => {
        let options = {
            mode: 'text',
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: 'src/py-scripts',
            args: []
        } as PythonShellOptions;
        const script = 'ACCOneTimeMemoryExtractor.py';

        try {
            //running the script in the main process (electron.js) instead this renderer process. we will wait for the result to comeback to onPythonMessage().
            const { shellId } = await window.electronAPI.runPythonScript(script, options);

            return new Promise((resolve, reject) => {
                let handleMessage = (returnedShellId: number, message: string) => {

                    if (shellId == returnedShellId) {//check valid session
                        try {
                            const obj = JSON.parse(message);

                            //if the script print out valid session map 
                            if (obj.Graphics.status == ACC_STATUS.ACC_LIVE) {
                                //find a valid live session, stop the checking process
                                stopCheckingLiveSessionInterval();
                                setValidLiveSession(true);
                                //set the static data too, so we can use it later.
                                analysisContext.setRecordedSessionStaticsData(obj);
                            }
                        } catch (error) {
                            // Handle the error appropriately
                        }
                    }

                }


                const handleScriptEnd = (returnedShellId: number) => {
                    if (shellId == returnedShellId) {// session recording is terminated
                        isCheckingLiveSession = false;
                        resolve("good");
                    }
                };

                setCheckSessionScriptShellId(shellId);


                // Set up listener for Python messages
                window.electronAPI.OnPythonMessageOnce(handleMessage);
                window.electronAPI.onPythonEnd(handleScriptEnd)

            })


        } catch (error) {
            isCheckingLiveSession = false;
        }
    };

    /**
     * run the python script and start record the session
     * @returns 
     */
    const StartRecording = async () => {

        //if no valid live sesssion, we dont do anything
        if (!hasValidLiveSession) return;

        const currentDate = new Date();
        const filename: string = `acc_${currentDate.getFullYear()}_${currentDate.getMonth()}_${currentDate.getDate()}_${currentDate.getHours()}_${currentDate.getMinutes()}_${currentDate.getSeconds()}.csv`;
        const folder: string = 'sessionRecording';

        let options = {
            mode: 'text',
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: 'src/py-scripts',
            args: [folder, filename]
        } as PythonShellOptions;
        const script = 'ACCMemoryExtractor.py';

        //find and set the track name by using the saved static data from the CheckSessionValid()

        if (!ACCMemoeryTracks.has(analysisContext.recordedSessioStaticsData.Static.track)) return;

        const trackname: string = ACCMemoeryTracks.get(analysisContext.recordedSessioStaticsData.Static.track)!
        analysisContext.setMap(trackname);
        analysisContext.setSession(new Date().toString());


        try {
            //running the script in the main process (electron.js) instead this renderer process
            const { shellId } = await window.electronAPI.runPythonScript(script, options);

            const offPythonMessage = window.electronAPI.onPythonMessage((incomingScriptShellId: number, message: string) => {

                if (shellId == incomingScriptShellId) { //check return result of recording script
                    try {
                        const obj = JSON.parse(message);
                        analysisContext.setLiveSessionData(obj);
                        analysisContext.setRecordedSessionData((presState: any) => {
                            return [...presState, obj];
                        });

                    } catch (error) {
                        // Handle the error appropriately
                    }
                }
            });

            window.electronAPI.onPythonEnd((incomingScriptShellId: number) => {

                if (shellId == incomingScriptShellId) {// session recording is terminated

                    setIsRecording(false);
                    setIsRecorEnded(true);
                    offPythonMessage();
                }
            })

            setIsRecording(true);

        } catch (error) {

        }
    };

    /**
     * start checking the valid live session in a interval
     */
    const startCheckingLiveSessionInterval = () => {


        if (!hasValidLiveSession && !isCheckingLiveSession) {
            //check every 1 sec by using a python script
            intervalRef.current = setInterval(CheckSessionValid, 2000);
        }
    };

    // Stop the interval
    const stopCheckingLiveSessionInterval = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    };

    async function handleUpload() {

        if (!analysisContext.options?.sessionOption || !analysisContext.options?.mapOption || !auth?.user) {
            return;
        }
        const data = analysisContext.recordedSessionData;;
        const chunks = [];
        const chunkSize = 10;
        const metadata = {
            sessionName: analysisContext.options.sessionOption,
            mapName: analysisContext.options.mapOption,
            userEmail: auth?.user,
        } as UploadReacingSessionInitDto;


        //seperate recored data into chunks
        for (let i = 0; i < data.length; i += chunkSize) {
            chunks.push(data.slice(i, i + chunkSize));
        }

        // First send metadata
        const initResponse = await apiService.post('/racing-session/upload/init', metadata);

        if (!initResponse.data) {
            throw new Error('First response missing required data');
        }
        const { uploadId } = initResponse.data as UploadReacingSessionInitReturnDto;

        // Then send chunks
        for (let i = 0; i < chunks.length; i++) {
            const url = '/racing-session/upload/chunk';
            const params = new URLSearchParams();
            params.append('uploadId', uploadId);
            const progress = await apiService.post(`${url}?${params.toString()}`, { chunk: chunks[i], chunkIndex: i });
        }

        // Finalize
        const url = '/racing-session/upload/complete';
        const params = new URLSearchParams();
        params.append('uploadId', uploadId);

        await apiService.post(`${url}?${params.toString()}`, {});
        reset();
        return true;
    }

    function reset() {
        setIsRecording(false);
        setIsRecorEnded(false);
        setValidLiveSession(false);
        setCheckSessionScriptShellId(-1);
        analysisContext.setSession(null);
    }

    return (

        <Box
            position="absolute"
            left="0"
            right="0"
            bottom="0"
            mb="5"
            height="64px"
            style={{
                borderRadius: "100px",
                boxShadow: "var(--shadow-6)",
                marginLeft: 200,
                marginRight: 200,
            }}
        >

            <Flex height="100%" justify="between" position="relative">
                <Flex gap="4" align="center" p="3">
                    {!isRecordEnded ?
                        //if record hasnt been initilized or stil recording
                        <IconButton radius="full" size="3" onClick={StartRecording}>
                            {hasValidLiveSession ?
                                (<svg xmlns="http://www.w3.org/2000/svg" fill="currentcolor" viewBox="0 0 30 30" width="20" height="20" style={{ marginRight: -2 }}>
                                    <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
                                </svg>
                                ) : (
                                    < Spinner size="3" />
                                )}
                        </IconButton> :

                        //record ended
                        <AlertDialog.Root>
                            <AlertDialog.Trigger>
                                <IconButton radius="full" size="3">
                                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7.81825 1.18188C7.64251 1.00615 7.35759 1.00615 7.18185 1.18188L4.18185 4.18188C4.00611 4.35762 4.00611 4.64254 4.18185 4.81828C4.35759 4.99401 4.64251 4.99401 4.81825 4.81828L7.05005 2.58648V9.49996C7.05005 9.74849 7.25152 9.94996 7.50005 9.94996C7.74858 9.94996 7.95005 9.74849 7.95005 9.49996V2.58648L10.1819 4.81828C10.3576 4.99401 10.6425 4.99401 10.8182 4.81828C10.994 4.64254 10.994 4.35762 10.8182 4.18188L7.81825 1.18188ZM2.5 9.99997C2.77614 9.99997 3 10.2238 3 10.5V12C3 12.5538 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2238 12.2239 9.99997 12.5 9.99997C12.7761 9.99997 13 10.2238 13 10.5V12C13 13.104 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2238 2.22386 9.99997 2.5 9.99997Z" fill="currentColor" fillRule="evenodd" clipRule="evenodd"></path></svg>
                                </IconButton>
                            </AlertDialog.Trigger>
                            <AlertDialog.Content maxWidth="450px">
                                <AlertDialog.Title>Revoke access</AlertDialog.Title>
                                <AlertDialog.Description size="2">
                                    Are you sure? This application will no longer be accessible and any
                                    existing sessions will be expired.
                                </AlertDialog.Description>


                                <Card size="4">

                                    <Heading as="h3" size="6" trim="start" mb="5">
                                        Session {" "}
                                        <Text as="div" size="3" weight="bold" color="blue">
                                            June 21, 2023
                                        </Text>
                                    </Heading>

                                    <Grid columns="2" gapX="4" gapY="5">
                                        <Box>
                                            <Text as="div" size="2" mb="1" color="gray">
                                                Started at
                                            </Text>
                                            <Text as="div" size="3" weight="bold">
                                                June 21, 2023
                                            </Text>
                                        </Box>

                                        <Box>
                                            <Text as="div" size="2" mb="1" color="gray">
                                                Ended at
                                            </Text>
                                            <Text as="div" size="3" weight="bold">
                                                July 21, 2023
                                            </Text>
                                        </Box>

                                        <Box>
                                            <Text as="div" size="2" mb="1" color="gray">
                                                Map
                                            </Text>
                                            <Text as="div" size="3" mb="1" weight="bold">
                                                Map name here
                                            </Text>
                                            <Text as="div" size="2">
                                                Pratice session
                                            </Text>
                                        </Box>


                                        <Flex direction="column" gap="1" gridColumn="1 / -1">

                                            <Flex justify="between">
                                                <Text size="3" mb="1" weight="bold">
                                                    Session length
                                                </Text>
                                                <Text size="2">00:20:00</Text>
                                            </Flex>
                                            <Flex justify="between">
                                                <Text size="3" mb="1" weight="bold">
                                                    Lap
                                                </Text>
                                                <Text size="2">4</Text>
                                            </Flex>
                                        </Flex>
                                    </Grid>
                                </Card>
                                <Flex gap="3" mt="4" justify="end">
                                    <AlertDialog.Cancel>
                                        <Button onClick={reset} variant="outline" color="red">
                                            Reject
                                        </Button>
                                    </AlertDialog.Cancel>
                                    <AlertDialog.Action>
                                        <Button onClick={handleUpload} >Approve</Button>
                                    </AlertDialog.Action>
                                </Flex>
                            </AlertDialog.Content>
                        </AlertDialog.Root>


                    }
                    <Flex align="center" gap="4">
                        <IconButton
                            color="gray"
                            variant="ghost"
                            radius="full"
                            size="2"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 30 30"
                                fill="currentcolor"
                                fillOpacity={0.7}
                                width="20"
                                height="20"
                            >
                                <path d="M 20 4 L 20 7 L 8 7 C 4.6983746 7 2 9.6983746 2 13 A 1.0001 1.0001 0 1 0 4 13 C 4 10.779625 5.7796254 9 8 9 L 20 9 L 20 12 L 27 8 L 20 4 z M 26.984375 15.986328 A 1.0001 1.0001 0 0 0 26 17 C 26 19.220375 24.220375 21 22 21 L 10 21 L 10 18 L 3 22 L 10 26 L 10 23 L 22 23 C 25.301625 23 28 20.301625 28 17 A 1.0001 1.0001 0 0 0 26.984375 15.986328 z" />
                            </svg>
                        </IconButton>
                    </Flex>
                </Flex>

                <Flex align="center" gap="3">

                    <Flex align="center" gap="3">
                        <Box>
                            <Text size="1" as="div" weight="medium">
                                Racing Map Name Here
                            </Text>
                            <Text size="1" as="div" color="gray" mb="2">
                                Partice Session
                            </Text>

                            <Box
                                position="relative"
                                height="4px"
                                width="320px"
                                style={{
                                    backgroundColor: "var(--gray-a5)",
                                    borderRadius: "var(--radius-1)",
                                }}
                            >
                                <Box
                                    position="absolute"
                                    height="4px"
                                    width="64px"
                                    style={{
                                        borderRadius: "var(--radius-1)",
                                        backgroundColor: "var(--gray-a9)",
                                    }}
                                />
                                <Box position="absolute" top="0" right="0" mt="-28px">
                                    <Text size="1" color="gray">
                                        0:58 / Lap 2
                                    </Text>
                                </Box>
                            </Box>
                        </Box>
                    </Flex>


                </Flex>

                <Flex align="center" gap="2" p="5">
                    <Slider
                        defaultValue={[80]}
                        variant="soft"
                        color="gray"
                        radius="full"
                        size="2"
                        style={{ width: 80 }}
                    />

                </Flex>
            </Flex>
        </Box>


    );

};







export default LiveAnalysisSessionRecording;