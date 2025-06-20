import { Card, Flex, Box, TextField, IconButton, Heading, Grid, Text, Slider, Avatar } from '@radix-ui/themes';
import { useContext, useEffect, useState } from 'react';

import { Link } from 'react-router-dom';
import { PythonShellOptions } from 'services/pythonService';
import { AnalysisContext } from '../session-analysis';
import { AllMapsBasicInfoListDto, MapOption } from 'data/live-analysis/live-analysis-data';
import apiService from 'services/api.service';


const LiveAnalysisSessionRecording = () => {
    const analysisContext = useContext(AnalysisContext);
    const [output, setOutput] = useState<string[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [scriptShellId, setScriptShellId] = useState(0);
    useEffect(() => {
        // Set up listener for Python messages
        window.electronAPI.onPythonMessage((shellId: number, message: string) => {

            if (shellId == scriptShellId) {
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
        window.electronAPI.onPythonEnd((shellId: number, message: string) => {
            if (shellId == scriptShellId) {
                apiService.get('/racingmap/map/infolists')
                    .then((result) => {
                        const data = result.data as AllMapsBasicInfoListDto;
                        let count = 0;

                    }).catch((e) => {
                    });
            }
        })
        return () => {
            // Clean up listeners when component unmounts
            window.electronAPI.onPythonMessage(() => { });
            window.electronAPI.onPythonEnd(() => { });
        };
    }, []);

    const runScript = async () => {
        let options = {
            mode: 'text',
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: 'src/py-scripts',
            args: []
        } as PythonShellOptions;
        const script = 'ACCMemoryExtractor.py';

        setIsRunning(true);
        setOutput([]);

        try {
            //running the script in the main process (electron.js) instead this renderer process
            const { shellId } = await window.electronAPI.runPythonScript(script, options);
            setScriptShellId(shellId);

        } catch (error) {
            setOutput(prev => [...prev, `Error: ${error}`]);
        } finally {
            setIsRunning(false);
        }
    };

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
                    <IconButton radius="full" size="3" onClick={runScript}>
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="currentcolor"
                            viewBox="0 0 30 30"
                            width="20"
                            height="20"
                            style={{ marginRight: -2 }}
                        >
                            <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
                        </svg>
                    </IconButton>

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